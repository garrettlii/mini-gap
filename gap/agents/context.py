"""Text every agent prompt shares: the workcell, the graph format, and one worked example.

FIREWALL: nothing here may describe how success is judged (gap/sim/success.py).
The worked example is a task outside the task set (hold a cube above a zone), so no
agent sees a solution to lift / place / stack / sort.
"""

from __future__ import annotations

WORKCELL = """\
A Franka Panda arm stands at the origin on a tabletop (z = 0). +x points away from the robot.
Three 4 cm cubes, colored red, green and blue, rest on the table in front of it
(about x 0.4-0.6 m, y -0.12-0.12 m). Two flat target zones: zone_a (yellow, at x 0.4, y +0.3)
and zone_b (cyan, at x 0.4, y -0.3), each 14 cm square. Unused cubes may sit far away.
Units: meters, radians. Poses come from perception skills; you cannot write Pose constants."""

EXAMPLE_INSTRUCTION = "pick up the blue cube and hold it above zone B"

EXAMPLE_GRAPH = """\
{"task": "hold_above",
 "nodes": [
  {"id": "find",   "skill": "locate_object_gt",   "in": {"color": "blue"}},
  {"id": "plan",   "skill": "plan_top_grasp",     "in": {"pose": "find.pose"}},
  {"id": "above",  "skill": "move_to_pose",       "in": {"pose": "plan.pre_pose"}},
  {"id": "open",   "skill": "open_gripper"},
  {"id": "reach",  "skill": "move_to_pose",       "in": {"pose": "plan.grasp_pose"}, "params": {"speed": 0.3}},
  {"id": "close",  "skill": "close_gripper"},
  {"id": "check",  "skill": "verify_grasp",       "in": {"color": "blue"}},
  {"id": "target", "skill": "compute_place_pose", "in": {"target": "zone_b"}},
  {"id": "carry",  "skill": "move_above",         "in": {"pose": "target.pose"}, "params": {"height": 0.15}}
 ],
 "edges": [{"from": "check", "to": "find", "on": "fail", "max_retries": 2}]}"""

SCHEMA_RULES = f"""\
A graph is one JSON object: {{"task": str, "nodes": [...], "edges": [...]}}.
- Each node: {{"id": lowercase_name, "skill": catalog name, "in": {{port: value}}, "params": {{name: number}}}}.
  Bind every input port the skill lists. Params are optional; stay inside their [lo, hi].
- An input value "<node_id>.<output_port>" refers to an output of an EARLIER node. Pose inputs
  must be such references. Other values (colors like "red", zones like "zone_a") are constants.
- Nodes run top to bottom. When a node succeeds, the next node in the list runs.
- When a node fails, the episode ends unless it has a fail edge:
  {{"from": node_id, "to": earlier_node_id, "on": "fail", "max_retries": 1-5}}.
- The last node ends the episode.

Example. Instruction: "{EXAMPLE_INSTRUCTION}"
{EXAMPLE_GRAPH}"""

EXAMPLE_CODE = """\
def run():
    for attempt in range(3):
        find = locate_object_gt(color="blue")
        plan = plan_top_grasp(pose=find["pose"])
        move_to_pose(pose=plan["pre_pose"])
        open_gripper()
        move_to_pose(pose=plan["grasp_pose"], speed=0.3)
        close_gripper()
        try:
            verify_grasp(color="blue")
            break
        except SkillFailed:
            continue
    target = compute_place_pose(target="zone_b")
    move_above(pose=target["pose"], height=0.15)"""

CODE_RULES = f"""\
Write Python: a function run() that calls skills as functions. Pass input ports and params as
keyword arguments. Each call returns a dict of the skill's outputs (e.g. find["pose"]) and raises
SkillFailed if the skill fails; catch it to retry. No imports; only plain Python and the skills.

Example. Instruction: "{EXAMPLE_INSTRUCTION}"
```python
{EXAMPLE_CODE}
```"""


def fill(template: str, **values: str) -> str:
    """Substitute {name} placeholders without str.format, so literal JSON braces survive."""
    for k, v in values.items():
        template = template.replace("{" + k + "}", v)
    return template
