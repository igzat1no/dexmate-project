"""
Coordinate resolver – converts semantic actions into executable commands.

The VLM planner outputs high-level semantic actions like:
    {"action": "place", "object_id_1": 2, "object_id_2": 1, "relationship": "next to"}

This module resolves them into concrete commands with 3D coordinates:
    {"action": "place", "object_id_1": 2, "object_id_2": 1,
     "position": [0.30, 0.07, 0.95], "relationship": "next to"}

Position logic:
  - pick up:  position = centroid of object_id_1
  - place:    position = centroid of reference + *orientation-aware* offset.

Orientation-aware offset:
  Each object has a `yaw` (PCA principal axis on the table plane).
  The table coordinate frame (u, v, normal) is stored in the table object.
  For the reference object, we reconstruct its forward and side axes
  from yaw + table frame, then use those for directional placement:
    - left_of / right_of:  along the reference's side axis
    - in_front_of / behind: along the reference's forward axis
    - on:                   along the table normal (upward)
    - next to / next_to:    along the reference's side axis (default)
    - in:                   zero offset (place at reference centroid)

  Both objects' OBB dimensions are projected onto the offset direction
  via the separating-axis theorem so that different orientations are
  handled correctly.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

DEFAULT_GAP = 0.02  # 2 cm clearance
DEFAULT_FALLBACK_HALF = 0.05  # 5 cm fallback half-extent


def _build_id_index(scene_json: list[dict]) -> dict[int, dict]:
    """Map object_id → object dict."""
    return {obj["object_id"]: obj for obj in scene_json}


def _extract_table_frame(
    scene_json: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract the table coordinate frame (u, v, normal) from the scene JSON.
    Falls back to camera-frame axes if the table is absent.
    """
    for obj in scene_json:
        if obj.get("label") == "table":
            u = np.array(obj.get("table_u_axis", [1, 0, 0]), dtype=np.float64)
            v = np.array(obj.get("table_v_axis", [0, 1, 0]), dtype=np.float64)
            n = np.array(obj.get("table_normal", [0, 0, -1]), dtype=np.float64)
            return u, v, n
    return (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, -1.0]),
    )


def _object_axes(
    obj: dict,
    table_u: np.ndarray,
    table_v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the 3D forward and side axes for an object from its yaw.

    forward = cos(yaw) * u + sin(yaw) * v   (object's main/long axis)
    side    = –sin(yaw) * u + cos(yaw) * v   (perpendicular on table)
    """
    yaw = obj.get("yaw", 0.0)
    c, s = np.cos(yaw), np.sin(yaw)
    forward = c * table_u + s * table_v
    side = -s * table_u + c * table_v
    return forward, side


def _projected_half_extent(
    obj: dict | None,
    direction: np.ndarray,
    table_u: np.ndarray,
    table_v: np.ndarray,
    table_normal: np.ndarray,
) -> float:
    """
    Project an object's OBB onto *direction* and return the half-extent.

    Uses the separating-axis theorem:
        half_extent = Σ_i ( half_size_i × |dot(obb_axis_i, direction)| )

    OBB axes for an object:
        axis 0 (length)  = object's forward  (from yaw)
        axis 1 (width)   = object's side     (from yaw)
        axis 2 (height)  = table normal
    """
    if obj is None:
        return DEFAULT_FALLBACK_HALF

    sz = obj.get("size_xyz")
    if not sz or len(sz) != 3 or max(sz) <= 0:
        return DEFAULT_FALLBACK_HALF

    half_l, half_w, half_h = sz[0] / 2.0, sz[1] / 2.0, sz[2] / 2.0

    fwd, side = _object_axes(obj, table_u, table_v)

    proj = (
        half_l * abs(float(np.dot(fwd, direction)))
        + half_w * abs(float(np.dot(side, direction)))
        + half_h * abs(float(np.dot(table_normal, direction)))
    )
    return max(proj, 0.001)


def _compute_place_offset(
    target_obj: dict | None,
    ref_obj: dict,
    relation: str,
    table_u: np.ndarray,
    table_v: np.ndarray,
    table_normal: np.ndarray,
    gap: float = DEFAULT_GAP,
) -> np.ndarray:
    """
    Compute the 3D offset from the reference centroid to the placement
    position, taking both objects' orientations and sizes into account.
    """
    forward, side = _object_axes(ref_obj, table_u, table_v)

    if relation in ("left_of",):
        direction = -side
    elif relation in ("right_of",):
        direction = side
    elif relation in ("in_front_of",):
        direction = forward
    elif relation in ("behind",):
        direction = -forward
    elif relation == "on":
        direction = table_normal
    elif relation == "in":
        return np.zeros(3)
    else:
        # "next to" / "next_to" / unknown → side direction
        direction = side

    ref_half = _projected_half_extent(
        ref_obj, direction, table_u, table_v, table_normal,
    )
    tgt_half = _projected_half_extent(
        target_obj, direction, table_u, table_v, table_normal,
    )

    distance = ref_half + tgt_half + gap
    return direction * distance


def resolve_plan(
    semantic_plan: list[dict],
    scene_json: list[dict],
    gap: float = DEFAULT_GAP,
) -> list[dict]:
    """
    Convert a semantic action plan into executable commands with positions.
    """
    index = _build_id_index(scene_json)
    table_u, table_v, table_normal = _extract_table_frame(scene_json)
    resolved: list[dict] = []

    for step in semantic_plan:
        action = step.get("action")

        if action == "done":
            resolved.append({"action": "done"})
            continue

        oid1 = step.get("object_id_1")

        if action == "pick up":
            obj = index.get(oid1)
            if obj is None:
                logger.warning("resolve: object_id_1={} not in scene, skipping", oid1)
                continue
            resolved.append({
                "action": "pick up",
                "object_id_1": oid1,
                "position": list(obj["centroid_xyz"]),
            })

        elif action == "place":
            oid2 = step.get("object_id_2")
            relation = step.get("relationship", "next to")

            ref_obj = index.get(oid2)
            target_obj = index.get(oid1)

            if ref_obj is None:
                logger.warning(
                    "resolve: object_id_2={} not in scene, using zero offset", oid2,
                )
                ref_pos = np.zeros(3)
                offset = np.zeros(3)
            else:
                ref_pos = np.array(ref_obj["centroid_xyz"])
                offset = _compute_place_offset(
                    target_obj, ref_obj, relation,
                    table_u, table_v, table_normal, gap,
                )

            place_pos = ref_pos + offset

            logger.info(
                "place obj {} {} obj {}: ref_pos={}, offset={} → pos={}",
                oid1, relation, oid2,
                [round(float(v), 4) for v in ref_pos],
                [round(float(v), 4) for v in offset],
                [round(float(v), 4) for v in place_pos],
            )

            resolved.append({
                "action": "place",
                "object_id_1": oid1,
                "object_id_2": oid2,
                "position": [round(float(v), 5) for v in place_pos],
                "relationship": relation,
            })

    return resolved
