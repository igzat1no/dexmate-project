#!/home/zongtai/miniconda3/envs/cl_cotnav/bin/python
"""
Language-Grounded Task Planner – Task 2.2

Two-stage pipeline:
  Stage 1 (VLM):      instruction → semantic action plan (no coordinates)
  Stage 2 (Resolver):  semantic plan + scene JSON → executable plan with positions

Usage (inside cl_cotnav conda env):
    # dry-run (no API call, demonstrates the full pipeline with an example):
    python -m task_planner.run --dry-run \\
        --scene-json output/003_004_024_025_061_f000_scene.json \\
        --instruction "Put the mug next to the bowl."

    # with a real API key (auto-discovers images from scene-json path):
    python -m task_planner.run --api-key YOUR_KEY \\
        --scene-json output/003_004_024_025_061_f000_scene.json \\
        --instruction "Put the mug next to the bowl."
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from loguru import logger

try:
    from .planner import TaskPlanner, SYSTEM_PROMPT, scene_to_vlm_format
    from .resolver import resolve_plan
    from .validator import validate_semantic_plan, validate_resolved_plan
except ImportError:
    _pkg_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.dirname(_pkg_dir) not in sys.path:
        sys.path.insert(0, os.path.dirname(_pkg_dir))
    from task_planner.planner import TaskPlanner, SYSTEM_PROMPT, scene_to_vlm_format
    from task_planner.resolver import resolve_plan
    from task_planner.validator import validate_semantic_plan, validate_resolved_plan

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")


def _auto_discover_images(scene_json_path: str):
    """
    Given a scene JSON path like output/003_004_024_025_061_f000_scene.json,
    auto-discover the corresponding _vis.jpg (raw) and _vlm.jpg (annotated).
    """
    base = scene_json_path.replace("_scene.json", "")
    vis_path = base + "_vis.jpg"
    vlm_path = base + "_vlm.jpg"
    return (
        vis_path if os.path.isfile(vis_path) else None,
        vlm_path if os.path.isfile(vlm_path) else None,
    )


def dry_run(scene_json: list[dict], instruction: str, image_path: str | None):
    """Simulate the full two-stage pipeline without calling the API."""
    id_to_label = {obj["object_id"]: obj["label"] for obj in scene_json}
    known_ids = sorted(id_to_label.keys())

    logger.info("=== DRY RUN (no API call) ===")
    logger.info("Known objects: {}", {k: id_to_label[k] for k in known_ids})
    logger.info("Instruction  : {}", instruction)
    logger.info("Image        : {}", image_path or "(none)")

    print("\n--- System Prompt (sent to Gemini) ---")
    print(SYSTEM_PROMPT)

    vlm_scene = scene_to_vlm_format(scene_json)
    print("\n--- Scene JSON (VLM format, first 3 objects) ---")
    print(json.dumps(vlm_scene[:3], indent=2))

    non_table = [o for o in scene_json if o["label"] != "table"]
    target_obj = next((o for o in non_table if "mug" in o["label"]), non_table[0] if non_table else scene_json[0])
    ref_obj = next((o for o in non_table if "bowl" in o["label"]),
                   non_table[1] if len(non_table) > 1 else non_table[0] if non_table else scene_json[0])

    semantic_plan = [
        {"action": "pick up", "object_id_1": target_obj["object_id"]},
        {"action": "place", "object_id_1": target_obj["object_id"],
         "object_id_2": ref_obj["object_id"], "relationship": "next_to"},
        {"action": "done"},
    ]

    print("\n--- Stage 1: Semantic Plan (VLM output, no coordinates) ---")
    print(json.dumps(semantic_plan, indent=2))

    sem_val = validate_semantic_plan(semantic_plan, set(id_to_label.keys()))
    print("\n--- Semantic Validation ---")
    print(json.dumps(sem_val, indent=2))

    resolved = resolve_plan(semantic_plan, scene_json)

    print("\n--- Stage 2: Resolved Executable Plan (with positions) ---")
    print(json.dumps(resolved, indent=2))

    res_val = validate_resolved_plan(resolved)
    print("\n--- Resolved Validation ---")
    print(json.dumps(res_val, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Task Planner – Task 2.2")
    parser.add_argument(
        "--api-key", type=str, default=os.environ.get("GEMINI_API_KEY", ""),
        help="Gemini API key (or set GEMINI_API_KEY env var)",
    )
    parser.add_argument("--model", type=str, default="gemini-2.5-flash")
    parser.add_argument(
        "--scene-json", type=str, required=True,
        help="Path to scene JSON from scene_understanding (Task 2.1)",
    )
    parser.add_argument(
        "--instruction", type=str, required=True,
        help='Natural-language command, e.g. "Put the mug next to the bowl."',
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Raw RGB image path (auto-discovered from scene-json if omitted)",
    )
    parser.add_argument(
        "--vlm-image", type=str, default=None,
        help="Annotated image path with object IDs (auto-discovered if omitted)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Don't call API")
    args = parser.parse_args()

    with open(args.scene_json) as f:
        scene_json = json.load(f)
    logger.info("Loaded {} objects from {}", len(scene_json), args.scene_json)

    # Auto-discover images if not explicitly provided
    auto_vis, auto_vlm = _auto_discover_images(args.scene_json)
    raw_image_path = args.image or auto_vis
    vlm_image_path = args.vlm_image or auto_vlm

    import cv2
    rgb = cv2.imread(raw_image_path, cv2.IMREAD_COLOR) if raw_image_path else None
    annotated = cv2.imread(vlm_image_path, cv2.IMREAD_COLOR) if vlm_image_path else None

    if rgb is not None:
        logger.info("Raw image     : {}", raw_image_path)
    if annotated is not None:
        logger.info("Annotated img : {}", vlm_image_path)

    if args.dry_run:
        dry_run(scene_json, args.instruction, raw_image_path)
        return

    if not args.api_key:
        logger.error(
            "No API key. Use --api-key KEY or set GEMINI_API_KEY. "
            "Or use --dry-run to preview."
        )
        sys.exit(1)

    # ── Stage 1: VLM semantic planning ──
    planner = TaskPlanner(api_key=args.api_key, model=args.model)
    result = planner.plan(
        scene_json, args.instruction,
        rgb_image=rgb,
        annotated_image=annotated,
    )

    print("\n=== Stage 1: Semantic Plan ===")
    print(json.dumps(result["plan"], indent=2))
    print("\n=== Reasoning ===")
    print(result["reasoning"])
    print("\n=== Semantic Validation ===")
    print(json.dumps(result["validation"], indent=2))

    if not result["validation"]["valid"]:
        logger.error("Semantic plan has errors, skipping resolution.")
        sys.exit(1)

    # ── Stage 2: Coordinate resolution ──
    resolved = resolve_plan(result["plan"], scene_json)
    res_val = validate_resolved_plan(resolved)

    print("\n=== Stage 2: Executable Plan ===")
    print(json.dumps(resolved, indent=2))
    print("\n=== Resolved Validation ===")
    print(json.dumps(res_val, indent=2))

    # save full output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "task_plan.json")
    full_result = {
        "instruction": args.instruction,
        "reasoning": result["reasoning"],
        "semantic_plan": result["plan"],
        "executable_plan": resolved,
        "semantic_validation": result["validation"],
        "resolved_validation": res_val,
    }
    with open(out_path, "w") as f:
        json.dump(full_result, f, indent=2)
    logger.info("Full plan saved → {}", out_path)


if __name__ == "__main__":
    main()
