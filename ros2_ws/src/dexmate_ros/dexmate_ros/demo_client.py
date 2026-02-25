#!/home/zongtai/miniconda3/envs/cl_cotnav/bin/python
"""
Demo client that calls /scene_understanding → /task_plan in sequence.

Usage:
    ros2 run dexmate_ros demo_client \
        --ros-args \
        -p rgb_path:=/path/to/000000.jpg \
        -p depth_path:=/path/to/000000.depth.png \
        -p instruction:="Put the mug next to the bowl." \
        -p api_key:=YOUR_KEY
"""

import json
import sys

import rclpy
from rclpy.node import Node

from dexmate_interfaces.srv import SceneUnderstanding, TaskPlan


class DemoClient(Node):
    def __init__(self):
        super().__init__("demo_client")

        self.declare_parameter("rgb_path", "")
        self.declare_parameter("depth_path", "")
        self.declare_parameter("instruction", "")
        self.declare_parameter("api_key", "")
        self.declare_parameter("scene_timeout", 120.0)
        self.declare_parameter("plan_timeout", 60.0)

        self.scene_cli = self.create_client(SceneUnderstanding, "scene_understanding")
        self.plan_cli = self.create_client(TaskPlan, "task_plan")

    def run(self):
        rgb_path = self.get_parameter("rgb_path").value
        depth_path = self.get_parameter("depth_path").value
        instruction = self.get_parameter("instruction").value
        api_key = self.get_parameter("api_key").value
        scene_timeout = self.get_parameter("scene_timeout").value
        plan_timeout = self.get_parameter("plan_timeout").value

        if not rgb_path or not depth_path or not instruction:
            self.get_logger().error(
                "Missing required params: rgb_path, depth_path, instruction"
            )
            return False

        # ── Stage 1: Scene Understanding ──
        self.get_logger().info("Waiting for /scene_understanding service ...")
        if not self.scene_cli.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("/scene_understanding service not available.")
            return False

        scene_req = SceneUnderstanding.Request()
        scene_req.rgb_path = rgb_path
        scene_req.depth_path = depth_path

        self.get_logger().info(f"Calling /scene_understanding ...")
        future = self.scene_cli.call_async(scene_req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=scene_timeout)

        scene_resp = future.result()
        if scene_resp is None:
            self.get_logger().error("Scene understanding service call timed out.")
            return False
        if not scene_resp.success:
            self.get_logger().error(f"Scene understanding failed: {scene_resp.message}")
            return False

        self.get_logger().info(f"Scene: {scene_resp.message}")
        scene_json_str = scene_resp.scene_json

        print("\n" + "=" * 60)
        print("SCENE JSON")
        print("=" * 60)
        scene_objs = json.loads(scene_json_str)
        for obj in scene_objs:
            print(f"  {obj['label']:20s}  centroid={obj['centroid_xyz']}  "
                  f"size={obj.get('size_xyz', 'N/A')}")

        # ── Stage 2: Task Planning ──
        self.get_logger().info("Waiting for /task_plan service ...")
        if not self.plan_cli.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("/task_plan service not available.")
            return False

        plan_req = TaskPlan.Request()
        plan_req.scene_json = scene_json_str
        plan_req.instruction = instruction
        plan_req.api_key = api_key

        self.get_logger().info(f"Calling /task_plan: \"{instruction}\"")
        future = self.plan_cli.call_async(plan_req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=plan_timeout)

        plan_resp = future.result()
        if plan_resp is None:
            self.get_logger().error("Task plan service call timed out.")
            return False
        if not plan_resp.success:
            self.get_logger().error(f"Task planning failed: {plan_resp.message}")
            return False

        self.get_logger().info(f"Plan: {plan_resp.message}")

        print("\n" + "=" * 60)
        print("VLM REASONING")
        print("=" * 60)
        print(plan_resp.reasoning)

        print("\n" + "=" * 60)
        print("SEMANTIC PLAN (from VLM)")
        print("=" * 60)
        print(plan_resp.semantic_plan_json)

        print("\n" + "=" * 60)
        print("EXECUTABLE PLAN (with coordinates)")
        print("=" * 60)
        print(plan_resp.executable_plan_json)

        return True


def main(args=None):
    rclpy.init(args=args)
    node = DemoClient()
    try:
        success = node.run()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
