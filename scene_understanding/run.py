#!/home/zongtai/miniconda3/envs/cl_cotnav/bin/python
"""
Scene Understanding pipeline – Task 2.1

Usage (inside cl_cotnav conda env):
    python -m scene_understanding.run                          # default test scene
    python -m scene_understanding.run --scene 004_005_008_025_037 --frame 5
    python -m scene_understanding.run --scene 003_004_024_025_061 --frame 0 --visualize
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
from loguru import logger

try:
    from .detector import GroundedSAMDetector
    from .point_cloud import (
        REALSENSE_R200,
        DEFAULT_DEPTH_SCALE,
        extract_object_point_clouds,
        export_scene_ply,
    )
    from .ycb_vocab import YCB_VOCAB
except ImportError:
    # Support direct execution: python scene_understanding/run.py
    _pkg_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.dirname(_pkg_dir) not in sys.path:
        sys.path.insert(0, os.path.dirname(_pkg_dir))
    from scene_understanding.detector import GroundedSAMDetector
    from scene_understanding.point_cloud import (
        REALSENSE_R200,
        DEFAULT_DEPTH_SCALE,
        extract_object_point_clouds,
        export_scene_ply,
    )
    from scene_understanding.ycb_vocab import YCB_VOCAB

# ── paths ──────────────────────────────────────────────────────────────
DATA_ROOT = "/home/zongtai/Project/Data/YCB-M/realsense_r200"
ANNO_ROOT = "/home/zongtai/Project/Data/YCB-M/realsense_r200_annotations/realsense_r200"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")

# ── test scenes (hand-picked for diversity) ────────────────────────────
DEFAULT_SCENES = [
    ("003_004_024_025_061", 0),   # cracker box, sugar box, bowl, mug, foam brick
    ("005_006_008_009_011_024", 3),  # tomato soup, mustard, pudding, gelatin, banana, bowl
    ("004_007_010_025_051", 2),   # sugar box, tuna, potted meat, mug, large clamp
]

_COLORMAP = None

def _label_colormap(n):
    """Generate n distinct colours (deterministic)."""
    global _COLORMAP
    if _COLORMAP is None or len(_COLORMAP) < n:
        rng = np.random.RandomState(42)
        _COLORMAP = rng.randint(60, 255, size=(max(n, 20), 3))
    return _COLORMAP[:n]


def load_rgbd(scene: str, frame: int):
    """Load one RGB-D pair from the YCB-M snapshot directory."""
    snap_dir = os.path.join(DATA_ROOT, scene, "snapshots")
    rgb_path = os.path.join(snap_dir, f"{frame:06d}.jpg")
    dep_path = os.path.join(snap_dir, f"{frame:06d}.depth.png")
    if not os.path.isfile(rgb_path):
        raise FileNotFoundError(rgb_path)
    if not os.path.isfile(dep_path):
        raise FileNotFoundError(dep_path)
    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    depth = cv2.imread(dep_path, cv2.IMREAD_UNCHANGED)
    return rgb, depth, rgb_path


def load_gt_annotations(scene: str, frame: int):
    """Load ground-truth annotation for comparison (optional)."""
    anno_path = os.path.join(
        ANNO_ROOT, scene, "snapshots", f"{frame:06d}.json"
    )
    if not os.path.isfile(anno_path):
        return None
    with open(anno_path) as f:
        return json.load(f)


def visualize(rgb, boxes, labels, confs, masks, save_path):
    """Draw detections + masks on the RGB image and save."""
    vis = rgb.copy()
    cmap = _label_colormap(len(labels))

    for i in range(len(labels)):
        color = tuple(int(c) for c in cmap[i])
        mask_i = masks[i].numpy() if hasattr(masks[i], "numpy") else masks[i]

        overlay = vis.copy()
        overlay[mask_i] = (
            np.array(overlay[mask_i], dtype=np.float32) * 0.5
            + np.array(color, dtype=np.float32) * 0.5
        ).astype(np.uint8)
        vis = overlay

        x1, y1, x2, y2 = [int(v) for v in boxes[i][:4]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        text = f"{labels[i]} {confs[i]:.2f}"
        cv2.putText(vis, text, (x1, max(y1 - 6, 0)),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    cv2.imwrite(save_path, vis)
    logger.info("Saved visualization → {}", save_path)


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

    Draws semi-transparent masks, bounding boxes, and "id=N label" tags
    inside each bounding box for explicit visual grounding.

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

        # Place the tag inside the bbox, at the top-left corner.
        # Clamp so the tag stays within the image.
        tx = max(x1 + 3, 0)
        ty = max(y1 + th + 4, th + 4)
        tx = min(tx, W - tw - 2)
        ty = min(ty, H - 2)

        # Dark background behind text for readability
        cv2.rectangle(
            vis,
            (tx - 2, ty - th - 3),
            (tx + tw + 2, ty + baseline + 1),
            color, cv2.FILLED,
        )
        cv2.putText(vis, tag, (tx, ty), font, font_scale,
                     (255, 255, 255), font_thickness, cv2.LINE_AA)

    cv2.imwrite(save_path, vis)
    logger.info("Saved VLM-annotated image → {}", save_path)
    return vis


def run_scene(detector: GroundedSAMDetector, scene: str, frame: int, do_vis: bool = True, output_dir: str = OUTPUT_DIR):
    """Full pipeline on one RGB-D frame."""
    logger.info("═" * 60)
    logger.info("Scene: {}  Frame: {}", scene, frame)
    logger.info("═" * 60)

    rgb, depth, rgb_path = load_rgbd(scene, frame)

    # ── detect + segment ──
    boxes, confs, labels, masks = detector.detect_and_segment(rgb)
    logger.info("Detected {} objects: {}", len(labels), list(labels))

    # ── extract 3-D point clouds from depth + masks ──
    masks_np = masks.numpy() if hasattr(masks, "numpy") else masks
    boxes_np = boxes.numpy() if hasattr(boxes, "numpy") else boxes
    confs_np = confs.numpy() if hasattr(confs, "numpy") else confs

    scene_json, raw_clouds = extract_object_point_clouds(
        depth=depth,
        masks=masks_np,
        labels=labels,
        confidences=confs_np,
        boxes=boxes_np,
        intrinsics=REALSENSE_R200,
        depth_scale=DEFAULT_DEPTH_SCALE,
    )

    # ── compare with GT annotations ──
    gt = load_gt_annotations(scene, frame)
    if gt is not None:
        gt_classes = [obj["class"] for obj in gt["objects"]]
        logger.info("GT objects in this frame: {}", gt_classes)

    # ── save outputs ──
    os.makedirs(output_dir, exist_ok=True)
    tag = f"{scene}_f{frame:03d}"

    json_path = os.path.join(output_dir, f"{tag}_scene.json")
    with open(json_path, "w") as f:
        json.dump(scene_json, f, indent=2)
    logger.info("Scene JSON → {}", json_path)

    # coloured point cloud
    ply_path = os.path.join(output_dir, f"{tag}_scene.ply")
    export_scene_ply(raw_clouds, ply_path)

    if do_vis and len(labels) > 0:
        vis_path = os.path.join(output_dir, f"{tag}_vis.jpg")
        visualize(rgb, boxes_np, labels, confs_np, masks_np, vis_path)

        # VLM-annotated image (with object_id tags inside bboxes)
        vlm_path = os.path.join(output_dir, f"{tag}_vlm.jpg")
        visualize_for_vlm(rgb, scene_json, masks_np, boxes_np, vlm_path)

    return scene_json


def main():
    parser = argparse.ArgumentParser(description="Scene Understanding – Task 2.1")
    parser.add_argument("--scene", type=str, default=None,
                        help="Scene folder name (e.g. 003_004_024_025_061)")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--visualize", action="store_true", default=True)
    parser.add_argument("--no-visualize", dest="visualize", action="store_false")
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    detector = GroundedSAMDetector(
        classes=YCB_VOCAB,
        box_threshold=args.box_threshold,
        device=args.device,
    )

    if args.scene:
        run_scene(detector, args.scene, args.frame, do_vis=args.visualize)
    else:
        for scene, frame in DEFAULT_SCENES:
            run_scene(detector, scene, frame, do_vis=args.visualize)


if __name__ == "__main__":
    main()
