#!/home/zongtai/miniconda3/envs/cl_cotnav/bin/python
"""
ROS 2 service node for Scene Understanding (Task 2.1).

Provides the /scene_understanding service:
  Request:  rgb_path, depth_path
  Response: scene_json (JSON string), success, message
"""

import json
import sys
import os

# Ensure the dexmate-project root is on PYTHONPATH so we can import
# scene_understanding / task_planner as regular packages.
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir)
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import cv2
import rclpy
from rclpy.node import Node
from loguru import logger

from dexmate_interfaces.srv import SceneUnderstanding

from scene_understanding.detector import GroundedSAMDetector
from scene_understanding.point_cloud import (
    REALSENSE_R200,
    DEFAULT_DEPTH_SCALE,
    extract_object_point_clouds,
)
from scene_understanding.ycb_vocab import YCB_VOCAB


class SceneUnderstandingNode(Node):
    """Service node that runs Grounding DINO + SAM + 3D lifting on demand."""

    def __init__(self):
        super().__init__("scene_understanding_node")

        self.declare_parameter("box_threshold", 0.25)
        self.declare_parameter("device", "cuda")

        box_thresh = self.get_parameter("box_threshold").value
        device = self.get_parameter("device").value

        self.get_logger().info(
            f"Loading detector (box_threshold={box_thresh}, device={device}) ..."
        )
        self.detector = GroundedSAMDetector(
            classes=YCB_VOCAB,
            box_threshold=box_thresh,
            device=device,
        )
        self.get_logger().info("Detector ready.")

        self.srv = self.create_service(
            SceneUnderstanding, "scene_understanding", self.handle_request,
        )
        self.get_logger().info("Service /scene_understanding is up.")

    def handle_request(self, request, response):
        rgb_path = request.rgb_path
        depth_path = request.depth_path
        self.get_logger().info(f"Request: rgb={rgb_path}  depth={depth_path}")

        if not os.path.isfile(rgb_path):
            response.success = False
            response.message = f"RGB file not found: {rgb_path}"
            return response
        if not os.path.isfile(depth_path):
            response.success = False
            response.message = f"Depth file not found: {depth_path}"
            return response

        try:
            rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

            boxes, confs, labels, masks = self.detector.detect_and_segment(rgb)

            masks_np = masks.numpy() if hasattr(masks, "numpy") else masks
            boxes_np = boxes.numpy() if hasattr(boxes, "numpy") else boxes
            confs_np = confs.numpy() if hasattr(confs, "numpy") else confs

            scene_json = extract_object_point_clouds(
                depth=depth,
                masks=masks_np,
                labels=labels,
                confidences=confs_np,
                boxes=boxes_np,
                intrinsics=REALSENSE_R200,
                depth_scale=DEFAULT_DEPTH_SCALE,
            )

            response.scene_json = json.dumps(scene_json)
            response.success = True
            response.message = f"Detected {len(scene_json)} objects."
            self.get_logger().info(response.message)

        except Exception as e:
            logger.exception("Scene understanding failed")
            response.success = False
            response.message = str(e)
            response.scene_json = "[]"

        return response


def main(args=None):
    rclpy.init(args=args)
    node = SceneUnderstandingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
