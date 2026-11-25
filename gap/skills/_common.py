"""Helpers shared by the skill modules (not skills themselves)."""

from __future__ import annotations

import numpy as np

from gap.sim import model_builder as mb
from gap.sim.ik import solve_ik
from gap.sim.motion import heading
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


def hand_offsets(cell) -> dict[str, float]:
    """The object nearest the hand, seen from the hand (a stand-in for a wrist camera, read
    from privileged state): its offset along the finger-closing axis and the finger-width
    axis, its height above the fingertip center, its yaw off square, and its half height
    and roundness (1 for the cylinder, whose yaw does not matter)."""
    state = cell.get_state()
    tcp = state["tcp"]
    name, obj = min(state["objects"].items(),
                    key=lambda kv: np.hypot(kv[1].pos[0] - tcp.pos[0], kv[1].pos[1] - tcp.pos[1]))
    variant = int(cell.instance.features.get(f"{name}_variant", 0)) if cell.instance else 0
    yaw = heading(tcp.quat)
    close_dir = yaw - np.pi / 2                      # fingers close along yaw - pi/2
    dx, dy = obj.pos[0] - tcp.pos[0], obj.pos[1] - tcp.pos[1]
    round_ = float(mb.VARIANTS[mb.VARIANT_NAMES[variant]][0] == "cylinder")
    dyaw = (heading(obj.quat) - yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
    return {"close_axis": float(dx * np.cos(close_dir) + dy * np.sin(close_dir)),
            "width_axis": float(-dx * np.sin(close_dir) + dy * np.cos(close_dir)),
            "dz": float(obj.pos[2] - tcp.pos[2]), "yaw": 0.0 if round_ else float(dyaw),
            "half": mb.half_height(variant), "round": round_, "object": name}
