#!/home/zongtai/miniconda3/envs/cl_cotnav/bin/python
"""
ROS 2 service node for Language-Grounded Task Planner (Task 2.2).

Provides the /task_plan service:
  Request:  scene_json, instruction, api_key
  Response: semantic_plan_json, executable_plan_json, reasoning, success, message
"""

import json
import sys
import os

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir)
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import rclpy
from rclpy.node import Node
from loguru import logger

from dexmate_interfaces.srv import TaskPlan

from task_planner.planner import TaskPlanner
from task_planner.resolver import resolve_plan
from task_planner.validator import validate_semantic_plan, validate_resolved_plan


class TaskPlannerNode(Node):
    """Service node that runs two-stage VLM planning on demand."""

    def __init__(self):
        super().__init__("task_planner_node")

        self.declare_parameter("model", "gemini-2.5-flash")
        self.declare_parameter("default_api_key", "")

        self.model = self.get_parameter("model").value
        self.default_api_key = self.get_parameter("default_api_key").value

        self._planner_cache: dict[str, TaskPlanner] = {}

        self.srv = self.create_service(TaskPlan, "task_plan", self.handle_request)
        self.get_logger().info(
            f"Service /task_plan is up  (model={self.model})."
        )

    def _get_planner(self, api_key: str) -> TaskPlanner:
        if api_key not in self._planner_cache:
            self._planner_cache[api_key] = TaskPlanner(
                api_key=api_key, model=self.model,
            )
        return self._planner_cache[api_key]

    def handle_request(self, request, response):
        instruction = request.instruction
        api_key = request.api_key or self.default_api_key
        self.get_logger().info(f"Request: \"{instruction}\"")

        if not api_key:
            response.success = False
            response.message = "No API key provided (set via request or default_api_key param)."
            return response

        try:
            scene_json = json.loads(request.scene_json)
        except json.JSONDecodeError as e:
            response.success = False
            response.message = f"Invalid scene_json: {e}"
            return response

        try:
            planner = self._get_planner(api_key)
            result = planner.plan(scene_json, instruction)

            semantic_plan = result["plan"]
            reasoning = result["reasoning"]
            validation = result["validation"]

            if not validation["valid"]:
                response.success = False
                response.message = f"Semantic plan invalid: {validation['errors']}"
                response.semantic_plan_json = json.dumps(semantic_plan)
                response.executable_plan_json = "[]"
                response.reasoning = reasoning
                return response

            executable_plan = resolve_plan(semantic_plan, scene_json)
            res_val = validate_resolved_plan(executable_plan)

            response.semantic_plan_json = json.dumps(semantic_plan, indent=2)
            response.executable_plan_json = json.dumps(executable_plan, indent=2)
            response.reasoning = reasoning
            response.success = True
            response.message = (
                f"Generated {len(executable_plan)} steps. "
                f"Resolved validation: {'OK' if res_val['valid'] else res_val['errors']}"
            )
            self.get_logger().info(response.message)

        except Exception as e:
            logger.exception("Task planning failed")
            response.success = False
            response.message = str(e)
            response.semantic_plan_json = "[]"
            response.executable_plan_json = "[]"
            response.reasoning = ""

        return response


def main(args=None):
    rclpy.init(args=args)
    node = TaskPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
