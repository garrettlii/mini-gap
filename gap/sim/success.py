"""Per-task success checks on privileged state.

FIREWALL: no agent prompt, skill, or LLM-visible text may import or quote this file.
Agents are graded by it; they must never be able to read it.
"""

from __future__ import annotations

import numpy as np

from gap.sim import model_builder as mb
from gap.sim.task_sampler import task_config
from gap.types import State

Z_TOL = 0.005          # meters: "resting" means within this of the expected height
OPEN_WIDTH = 0.045     # meters: wider than a 4 cm cube, so the gripper has let go
REST_STEPS = 100       # 0.2 s hold for the at-rest check
REST_TOL = 1e-3        # meters an object may drift during the hold


def check(task: str, state: State, task_cfg: dict, half: dict[str, float] | None = None) -> bool:
    """Dispatch to the per-task check using task_cfg['success'] from configs/tasks.yaml.

    lift:  object z - CUBE_HALF >= min_height
    place: object center inside the zone's xy square, resting (z near CUBE_HALF),
           and gripper open if required
    stack, invert: top cube xy within xy_tol of bottom cube, and z about one cube height higher
    sort, unstack:  every object in its assigned zone
    Also require objects to be at rest (check velocity, or step a few times and compare);
    a cube sliding through the zone is not a success.

    This is the static part; settled_success() adds the at-rest check. half maps object ->
    its true half height (object split variants); default CUBE_HALF.
    """
    c = task_cfg["success"]
    obj = state["objects"]
    h = {o: (half or {}).get(o, mb.CUBE_HALF) for o in obj}
    if task == "lift":
        return obj[c["object"]].pos[2] - h[c["object"]] >= c["min_height"]
    if task == "place":
        opened = state["gripper_width"] >= OPEN_WIDTH or not c.get("gripper_open", False)
        return _resting_in_zone(obj[c["object"]].pos, c["zone"], h[c["object"]]) and opened
    if task in ("stack", "invert"):
        top, bottom = obj[c["top"]].p, obj[c["bottom"]].p
        return (np.linalg.norm(top[:2] - bottom[:2]) <= c["xy_tol"]
                and abs(top[2] - bottom[2] - h[c["top"]] - h[c["bottom"]]) <= Z_TOL
                and abs(bottom[2] - h[c["bottom"]]) <= Z_TOL)
    if task in ("sort", "unstack"):
        return all(_resting_in_zone(obj[o].pos, z, h[o]) for o, z in c["assign"].items())
    raise ValueError(f"unknown task {task!r}")


def settled_success(cell) -> bool:
    """check() on the cell's task, after holding the controls REST_STEPS and seeing no drift."""
    task = cell.instance.task
    before = cell.get_state()["objects"]
    after = cell.step(None, REST_STEPS)
    still = all(np.linalg.norm(after["objects"][k].p - p.p) <= REST_TOL for k, p in before.items())
    half = {c: mb.half_height(int(cell.instance.features.get(f"{c}_variant", 0)))
            for c in mb.CUBES}
    return still and check(task, after, task_config()["tasks"][task], half)


def _resting_in_zone(pos, zone: str, half: float = mb.CUBE_HALF) -> bool:
    z = task_config()["zones"][zone]
    return (abs(pos[0] - z["pos"][0]) <= z["half"] and abs(pos[1] - z["pos"][1]) <= z["half"]
            and abs(pos[2] - half) <= Z_TOL)
