"""
Language-Grounded Task Planner – Task 2.2  (semantic layer)

The VLM is responsible ONLY for semantic reasoning:
  - Which object to pick (by object_id)
  - Where to place it (expressed as a spatial relation to another object_id)
  - Whether blockers need to be moved first

It does NOT generate any 3D coordinates.  A separate resolver module
(resolver.py) converts the semantic plan into executable commands
with concrete positions looked up from the scene JSON.

Gemini is accessed through the openai Python SDK with a custom base_url,
following the same pattern as VLM_ROS/vlm_node.
"""

from __future__ import annotations

import base64
import json
from typing import Optional

import cv2
import numpy as np
from loguru import logger
from openai import OpenAI
from pydantic import BaseModel

try:
    from .validator import validate_semantic_plan
except ImportError:
    import os, sys
    _pkg_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.dirname(_pkg_dir) not in sys.path:
        sys.path.insert(0, os.path.dirname(_pkg_dir))
    from task_planner.validator import validate_semantic_plan

# ── defaults ───────────────────────────────────────────────────────────
DEFAULT_MODEL = "gemini-2.5-flash"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


# ── structured output schemas (semantic only, no coordinates) ──────────
class SemanticAction(BaseModel):
    action: str                            # "pick up" | "place" | "done"
    object_id_1: Optional[int] = None      # object to manipulate
    object_id_2: Optional[int] = None      # reference object (place only)
    relationship: Optional[str] = None     # "on" | "in" | "next to" | ...


class PlanResult(BaseModel):
    reasoning: str
    plan: list[SemanticAction]


# ── system prompt ──────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are highly skilled in robotic task planning, breaking down intricate and long-term tasks into distinct primitive actions. You need to perform a tabletop pick-and-place task.

If the object is in sight, you need to directly manipulate it. If the object is not in sight, you need to use primitive skills to find the object first. If the target object is blocked by other objects, you need to remove all the blocking objects before picking up the target object. At the same time, you need to ignore distractors that are not related to the task. And remember your last step plan needs to be "done".

You must strictly follow the rules below.

Available Skills (ONLY these two):
1. pick up <object_id>
2. place <object_id> in/on/next_to/in_front_of/behind/left_of/right_of <object_id>

You are only allowed to use the provided skills.

Input Format:

You will receive the following inputs:

1. **Raw image**: An unmodified photo showing the current state of the objects on the table.
2. **Annotated image**: The same scene overlaid with bounding boxes, segmentation masks, and object ID labels (e.g. "id=3 cracker box"). Use this to match each `object_id` in the JSON to the corresponding physical object.
3. **Scene JSON**: A structured description of every detected object (see schema below).
4. **Instruction**: A natural language command describing the task.

The Scene JSON contains the following fields per object:

- `object_id`: A unique integer identifier for the object.
- `class`: The category of the object (e.g. "mug", "cracker box").
- `position`: The 3D centroid coordinates [x, y, z] in the camera frame (metres).
- `confidence`: Detection confidence score.
- `size`: Oriented bounding box dimensions [length, width, height] in metres.
- `yaw`: Orientation angle (radians) of the object's main axis on the table surface.

You must carefully analyze the images and the JSON to convert the natural language instruction into a sequence of primitive robot actions.

Output Format:

Your response should be a list. Each element in the list must be one of the following two formats:

If action is "pick up":
{
  "action": "pick up",
  "object_id_1": <integer>
}

If action is "place":
{
  "action": "place",
  "object_id_1": <integer>,
  "object_id_2": <integer>,
  "relationship": "on" | "in" | "next_to" | "in_front_of" | "behind" | "left_of" | "right_of"
}

The final element in the array MUST be:
{
  "action": "done"
}

Additional Rules:

1. All object IDs must exist in the provided JSON.
2. Do NOT invent new object IDs.
3. Do NOT output any fields other than those specified above.
4. Use the annotated image to verify which physical object corresponds to each object_id.

Think step-by-step in the "reasoning" field, then output the "plan" array.
"""


def scene_to_vlm_format(scene_json: list[dict]) -> list[dict]:
    """
    Transform internal scene JSON to the format described in the VLM prompt.

    Internal:  object_id, label, centroid_xyz, size_xyz, confidence, bbox, num_points, yaw
    VLM:       object_id, class, position, size, confidence, yaw
    """
    vlm_objects = []
    for obj in scene_json:
        vlm_objects.append({
            "object_id": obj["object_id"],
            "class": obj["label"],
            "position": obj["centroid_xyz"],
            "confidence": obj["confidence"],
            "size": obj["size_xyz"],
            "yaw": round(obj.get("yaw", 0.0), 3),
        })
    return vlm_objects


def _encode_image(img: np.ndarray) -> str:
    """Encode a BGR numpy image as a base64 JPEG string."""
    jpg = cv2.imencode(".jpg", img)[1]
    return base64.b64encode(jpg).decode("utf-8")


class TaskPlanner:
    """Gemini-based language-grounded task planner (semantic layer)."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = GEMINI_BASE_URL,
    ):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        logger.info("TaskPlanner initialised  model={}", model)

    def plan(
        self,
        scene_json: list[dict],
        instruction: str,
        rgb_image: Optional[np.ndarray] = None,
        annotated_image: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Generate a *semantic* action plan (no coordinates).

        Args:
            scene_json:      Scene object list from scene_understanding.
            instruction:     Natural language task command.
            rgb_image:       Raw RGB image (BGR, numpy).
            annotated_image: Image with bbox/mask/object_id annotations.

        Returns dict with keys: "reasoning", "plan", "validation".
        """
        vlm_scene = scene_to_vlm_format(scene_json)

        user_content: list[dict] = []

        # Image 1: raw image
        if rgb_image is not None:
            user_content.append({"type": "text", "text": "Raw image:"})
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{_encode_image(rgb_image)}"},
            })

        # Image 2: annotated image with object_id labels
        if annotated_image is not None:
            user_content.append({"type": "text", "text": "Annotated image (with object IDs):"})
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{_encode_image(annotated_image)}"},
            })

        user_content.append({
            "type": "text",
            "text": f"Scene JSON:\n```json\n{json.dumps(vlm_scene, indent=2)}\n```",
        })
        user_content.append(
            {"type": "text", "text": f"Instruction: {instruction}"}
        )

        logger.info("Calling Gemini ({}) – {}", self.model, instruction)

        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            response_format=PlanResult,
        )

        result: PlanResult = completion.choices[0].message.parsed
        logger.info("Gemini reasoning: {}", result.reasoning[:200])

        plan_dicts = [step.model_dump(exclude_none=True) for step in result.plan]

        known_ids = {obj["object_id"] for obj in scene_json}
        validation = validate_semantic_plan(plan_dicts, known_ids)

        return {
            "reasoning": result.reasoning,
            "plan": plan_dicts,
            "validation": validation,
        }
