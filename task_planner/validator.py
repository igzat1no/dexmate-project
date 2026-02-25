"""
Validation for the two-stage task planner.

Stage 1 – validate_semantic_plan():
    Checks the VLM's semantic output (no coordinates expected).

Stage 2 – validate_resolved_plan():
    Checks the resolved executable plan (coordinates present).
"""

from __future__ import annotations

from loguru import logger

VALID_ACTIONS = {"pick up", "place", "done"}
VALID_RELATIONS = {
    "next to", "next_to", "on", "in", "in_front_of", "behind", "left_of", "right_of",
}


def validate_semantic_plan(
    plan: list[dict],
    known_ids: set[int],
) -> dict:
    """
    Validate the semantic plan produced by the VLM.

    Checks:
      - Every referenced object_id exists in the scene.
      - Plan ends with 'done'.
      - 'place' actions have a valid relationship and object_id_2.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not plan:
        errors.append("Plan is empty.")
        return {"valid": False, "errors": errors, "warnings": warnings}

    for i, step in enumerate(plan):
        action = step.get("action")

        if action not in VALID_ACTIONS:
            errors.append(f"Step {i}: unknown action '{action}'.")
            continue

        if action == "done":
            if i != len(plan) - 1:
                warnings.append(
                    f"Step {i}: 'done' is not the last step; "
                    "trailing steps will be ignored."
                )
            continue

        oid1 = step.get("object_id_1")
        if oid1 is None:
            errors.append(f"Step {i} ({action}): missing 'object_id_1'.")
        elif oid1 not in known_ids:
            errors.append(
                f"Step {i} ({action}): object_id_1={oid1} not in scene. "
                f"Known: {sorted(known_ids)}"
            )

        if action == "place":
            oid2 = step.get("object_id_2")
            rel = step.get("relationship")

            if oid2 is None:
                errors.append(f"Step {i} (place): missing 'object_id_2'.")
            elif oid2 not in known_ids:
                errors.append(
                    f"Step {i} (place): object_id_2={oid2} not in scene."
                )

            if not rel:
                warnings.append(
                    f"Step {i} (place): missing 'relationship', "
                    "will default to 'next to'."
                )
            elif rel not in VALID_RELATIONS:
                warnings.append(
                    f"Step {i} (place): unknown relationship '{rel}'. "
                    f"Valid: {sorted(VALID_RELATIONS)}"
                )

    if plan[-1].get("action") != "done":
        errors.append("Plan does not end with 'done'.")

    valid = len(errors) == 0
    for e in errors:
        logger.warning("Semantic validation error: {}", e)
    for w in warnings:
        logger.info("Semantic validation warning: {}", w)

    return {"valid": valid, "errors": errors, "warnings": warnings}


def validate_resolved_plan(plan: list[dict]) -> dict:
    """
    Validate the resolved plan (post coordinate-resolution).

    Checks that pick/place steps have well-formed [x, y, z] positions.
    """
    errors: list[str] = []
    warnings: list[str] = []

    for i, step in enumerate(plan):
        action = step.get("action")
        if action == "done":
            continue

        pos = step.get("position")
        if pos is None:
            errors.append(f"Step {i} ({action}): missing 'position' after resolution.")
        elif not (isinstance(pos, list) and len(pos) == 3):
            errors.append(f"Step {i} ({action}): 'position' must be [x, y, z].")

    valid = len(errors) == 0
    for e in errors:
        logger.warning("Resolved validation error: {}", e)

    return {"valid": valid, "errors": errors, "warnings": warnings}
