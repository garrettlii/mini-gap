"""Helpers shared by the skill modules (not skills themselves)."""

from __future__ import annotations

import numpy as np

from gap.sim import model_builder as mb
from gap.sim.ik import solve_ik
from gap.types import Pose

GRIPPER_BODIES = ("hand", "left_finger", "right_finger")
V_MAX = 1.5        # rad/s joint speed at speed = 1.0
MIN_MOVE_S = 0.5   # shortest ramp, so small moves are still smooth


def cube_body(color: str) -> str | None:
    body = f"{color}_cube"
    return body if body in mb.CUBES else None


def reachable(cell, pose: Pose) -> tuple[bool, float]:
    """(IK converged, position residual in meters) from the current command."""
    ik = solve_ik(cell.model, cell.data, pose, q_init=cell.data.ctrl[:7].copy())
    return ik.converged, ik.pos_err


def move_duration(cell, q: np.ndarray, speed: float) -> float:
    dq = float(np.max(np.abs(q - cell.data.ctrl[:7])))
    return max(MIN_MOVE_S, dq / (speed * V_MAX))


def touching(state, body: str, others=GRIPPER_BODIES) -> set[str]:
    """Which of `others` are in contact with `body`."""
    return {b for pair in state["contacts"] for b in pair if body in pair and b in others}
