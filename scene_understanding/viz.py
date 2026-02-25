"""
Visualization helpers for scene understanding.
Shared by run.py and scene_worker.
"""

import numpy as np
import cv2

_COLORMAP = None


def _label_colormap(n):
    """Generate n distinct colours (deterministic)."""
    global _COLORMAP
    if _COLORMAP is None or len(_COLORMAP) < n:
        rng = np.random.RandomState(42)
        _COLORMAP = rng.randint(60, 255, size=(max(n, 20), 3))
    return _COLORMAP[:n]


def _match_detection_idx(scene_bbox: list, boxes_np: np.ndarray) -> int | None:
    """Find the detection index whose bbox matches the scene_json entry."""
    for j in range(len(boxes_np)):
        box_j = [round(float(b), 1) for b in boxes_np[j][:4]]
        if box_j == scene_bbox:
            return j
    return None


def visualize_for_vlm(
    rgb: np.ndarray,
    scene_json: list[dict],
    masks_np: np.ndarray,
    boxes_np: np.ndarray,
    save_path: str,
    mask_alpha: float = 0.35,
    font_scale: float = 0.55,
    font_thickness: int = 2,
) -> np.ndarray:
    """
    Generate an annotated image for VLM input.
    Draws semi-transparent masks, bounding boxes, and "id=N label" tags.
    Returns the annotated image (also saved to save_path).
    """
    H, W = rgb.shape[:2]
    vis = rgb.copy()
    cmap = _label_colormap(20)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for obj in scene_json:
        if obj["label"] == "table":
            continue

        oid = obj["object_id"]
        lbl = obj["label"]
        bbox = obj["bbox"]
        color = tuple(int(c) for c in cmap[oid % len(cmap)])

        det_idx = _match_detection_idx(bbox, boxes_np)
        if det_idx is not None:
            mask_i = masks_np[det_idx]
            if hasattr(mask_i, "numpy"):
                mask_i = mask_i.numpy()
            overlay = vis.copy()
            overlay[mask_i] = (
                np.array(overlay[mask_i], dtype=np.float32) * (1 - mask_alpha)
                + np.array(color, dtype=np.float32) * mask_alpha
            ).astype(np.uint8)
            vis = overlay

        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        tag = f"id={oid} {lbl}"
        (tw, th), baseline = cv2.getTextSize(tag, font, font_scale, font_thickness)

        tx = max(x1 + 3, 0)
        ty = max(y1 + th + 4, th + 4)
        tx = min(tx, W - tw - 2)
        ty = min(ty, H - 2)

        cv2.rectangle(
            vis,
            (tx - 2, ty - th - 3),
            (tx + tw + 2, ty + baseline + 1),
            color, cv2.FILLED,
        )
        cv2.putText(vis, tag, (tx, ty), font, font_scale,
                     (255, 255, 255), font_thickness, cv2.LINE_AA)

    cv2.imwrite(save_path, vis)
    return vis
