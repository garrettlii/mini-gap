"""Per-task success checks on privileged state.

FIREWALL: no agent prompt, skill, or LLM-visible text may import or quote this file.
Agents are graded by it; they must never be able to read it.
"""

from __future__ import annotations

from gap.types import State


def check(task: str, state: State, task_cfg: dict) -> bool:
    """Dispatch to the per-task check using task_cfg['success'] from configs/tasks.yaml.

    lift:  object z - CUBE_HALF >= min_height
    place: object center inside the zone's xy square, resting (z near CUBE_HALF),
           and gripper open if required
    stack: top cube xy within xy_tol of bottom cube, and z about one cube height higher
    sort:  every object in its assigned zone
    Also require objects to be at rest (check velocity, or step a few times and compare);
    a cube sliding through the zone is not a success.
    """
    raise NotImplementedError
