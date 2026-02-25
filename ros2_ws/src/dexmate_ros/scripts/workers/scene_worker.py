#!/home/zongtai/miniconda3/envs/cl_cotnav/bin/python
"""
Subprocess worker for scene understanding.
Runs in conda Python with ML dependencies.

Input (argv):  rgb_path  depth_path  [box_threshold]  [device]
Output (stdout): JSON with scene_json, raw_image_path, annotated_image_path, point_cloud_path
"""

import json
import sys
import os

# Redirect stdout to stderr during imports and inference to prevent
# mmengine/mmdet/loguru from contaminating the JSON output.
_real_stdout = sys.stdout
sys.stdout = sys.stderr

_PROJECT_ROOT = os.environ.get(
    "DEXMATE_PROJECT_ROOT",
    "/home/zongtai/Project/Codes/dexmate-project",
)
sys.path.insert(0, _PROJECT_ROOT)

import cv2
from scene_understanding.detector import GroundedSAMDetector
from scene_understanding.point_cloud import (
    REALSENSE_R200,
    DEFAULT_DEPTH_SCALE,
    extract_object_point_clouds,
    export_scene_ply,
)
from scene_understanding.viz import visualize_for_vlm
from scene_understanding.ycb_vocab import YCB_VOCAB


def main():
    if len(sys.argv) < 3:
        print("Usage: scene_worker.py <rgb_path> <depth_path> [box_threshold] [device]",
              file=sys.stderr)
        sys.exit(1)

    rgb_path = sys.argv[1]
    depth_path = sys.argv[2]
    box_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
    device = sys.argv[4] if len(sys.argv) > 4 else "cuda"

    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if rgb is None or depth is None:
        print(f"Failed to read images: {rgb_path}, {depth_path}", file=sys.stderr)
        sys.exit(1)

    detector = GroundedSAMDetector(
        classes=YCB_VOCAB,
        box_threshold=box_threshold,
        device=device,
    )
    boxes, confs, labels, masks = detector.detect_and_segment(rgb)

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

    # Save annotated image and colored point cloud for RViz visualization
    base_dir = os.path.dirname(rgb_path)
    base_name = os.path.splitext(os.path.basename(rgb_path))[0]
    annotated_path = os.path.join(base_dir, f"{base_name}_annotated.jpg")
    ply_path = os.path.join(base_dir, f"{base_name}_scene.ply")

    visualize_for_vlm(rgb, scene_json, masks_np, boxes_np, annotated_path)
    export_scene_ply(raw_clouds, ply_path)

    result = {
        "scene_json": scene_json,
        "raw_image_path": rgb_path,
        "annotated_image_path": annotated_path,
        "point_cloud_path": ply_path,
    }
    _real_stdout.write(json.dumps(result))
    _real_stdout.flush()


if __name__ == "__main__":
    main()
