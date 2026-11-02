"""Hand-scripted oracle for Place, using ground-truth poses.

The oracle proves the physics works and sets the ceiling for every later method.
Phase 1 is done when this succeeds on >= 95 of 100 dev seeds.

Sequence: move above cube -> descend -> close -> lift -> move above zone -> lower -> open
-> retreat. Use solve_ik for each waypoint and interpolate joint targets over a fixed
number of steps (a jump straight to the IK solution will fling the cube).
"""

from __future__ import annotations

from gap.sim.workcell import Workcell
from gap.types import TaskInstance


def run_place_oracle(cell: Workcell, instance: TaskInstance, record: bool = False) -> bool:
    """Return success. If record=True, also append frames to a list you can save as MP4."""
    raise NotImplementedError
