"""Hand-scripted oracle for Place, using ground-truth poses.

The oracle proves the physics works and sets the ceiling for every later method.
Phase 1 is done when this succeeds on >= 95 of 100 dev seeds.

Sequence: move above cube -> descend -> close -> lift -> move above zone -> lower -> open
-> retreat. Use solve_ik for each waypoint and interpolate joint targets over a fixed
number of steps (a jump straight to the IK solution will fling the cube).
"""

from __future__ import annotations

from gap.sim import model_builder as mb
from gap.sim.motion import grasp_quat, move_tcp, set_gripper
from gap.sim.success import settled_success
from gap.sim.task_sampler import task_config
from gap.sim.workcell import Workcell
from gap.types import Pose, TaskInstance

HOVER = 0.12       # TCP height above the table for transit, meters
GRASP_Z = 0.02     # TCP height at grasp: fingertip center at the cube center
RELEASE_Z = 0.03   # TCP height at release: cube bottom ~1 cm above the zone


def run_place_oracle(cell: Workcell, instance: TaskInstance,
                     frame_sink: list | None = None) -> bool:
    """Return success. Frames from the front camera go to frame_sink if given."""
    cell.frame_sink = frame_sink
    s = cell.reset(instance)
    cube = s["objects"]["red_cube"]
    q = grasp_quat(cube.quat, s["tcp"].quat)
    (cx, cy, _), (zx, zy, _) = cube.pos, task_config()["zones"]["zone_a"]["pos"]
    path = [((cx, cy, HOVER), 1.5), ((cx, cy, GRASP_Z), 1.0), "close",
            ((cx, cy, HOVER), 1.0), ((zx, zy, HOVER), 1.5), ((zx, zy, RELEASE_Z), 1.0), "open",
            ((zx, zy, HOVER), 1.0)]
    for step in path:
        if isinstance(step, str):
            set_gripper(cell, step == "open")
        elif not move_tcp(cell, Pose(step[0], q), step[1]).converged:
            return False
    return settled_success(cell)


assert mb.CUBE_HALF == GRASP_Z, "GRASP_Z assumes the 4 cm cube"
