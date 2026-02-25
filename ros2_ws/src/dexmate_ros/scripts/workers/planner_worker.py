#!/home/zongtai/miniconda3/envs/cl_cotnav/bin/python
"""
Subprocess worker for task planning.
Runs in conda Python with openai/pydantic dependencies.

Input (stdin):  JSON {"scene_json": [...], "instruction": "...", "api_key": "...", "model": "..."}
Output (stdout): JSON {"semantic_plan": [...], "executable_plan": [...], ...}
"""

import json
import sys
import os

_real_stdout = sys.stdout
sys.stdout = sys.stderr

_PROJECT_ROOT = os.environ.get(
    "DEXMATE_PROJECT_ROOT",
    "/home/zongtai/Project/Codes/dexmate-project",
)
sys.path.insert(0, _PROJECT_ROOT)

from task_planner.planner import TaskPlanner
from task_planner.resolver import resolve_plan
from task_planner.validator import validate_semantic_plan, validate_resolved_plan


def main():
    raw = sys.stdin.read()
    try:
        req = json.loads(raw)
    except json.JSONDecodeError as e:
        result = {"success": False, "message": f"Invalid JSON input: {e}"}
        _real_stdout.write(json.dumps(result))
        sys.exit(1)

    scene_json = req.get("scene_json", [])
    instruction = req.get("instruction", "")
    api_key = req.get("api_key", "")
    model = req.get("model", "gemini-2.5-flash")

    if not api_key:
        _real_stdout.write(json.dumps({"success": False, "message": "No API key"}))
        sys.exit(1)

    try:
        planner = TaskPlanner(api_key=api_key, model=model)
        result = planner.plan(scene_json, instruction)

        semantic_plan = result["plan"]
        validation = result["validation"]

        if not validation["valid"]:
            _real_stdout.write(json.dumps({
                "success": False,
                "message": f"Semantic plan invalid: {validation['errors']}",
                "semantic_plan": semantic_plan,
                "executable_plan": [],
                "reasoning": result["reasoning"],
            }))
            return

        executable_plan = resolve_plan(semantic_plan, scene_json)

        _real_stdout.write(json.dumps({
            "success": True,
            "message": f"Generated {len(executable_plan)} steps.",
            "semantic_plan": semantic_plan,
            "executable_plan": executable_plan,
            "reasoning": result["reasoning"],
        }))

    except Exception as e:
        _real_stdout.write(json.dumps({"success": False, "message": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
