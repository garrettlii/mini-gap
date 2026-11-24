"""Joint-space motion on the Workcell: IK a TCP pose, then ramp the servo targets to it.

Shared by the Place oracle (Phase 1) and the motion/grasp skills (Phase 2). Jumping the
position servos straight to an IK solution whips the arm and flings a held cube, so every
move ramps ctrl along a smoothstep from the current command to the goal.
"""

from __future__ import annotations

import mujoco
import numpy as np

from gap.sim import model_builder as mb
from gap.sim.ik import IKResult, solve_ik, top_down_quat
from gap.sim.workcell import Workcell
from gap.types import Pose, State

CHUNK = 10          # sim steps per ramp increment (20 ms at dt = 2 ms)
MIN_HOLD_S, MAX_HOLD_S = 0.1, 1.0   # hold at the goal so the servos catch up
SETTLE_RAD = 1e-3   # per-joint tracking error that counts as arrived


def move_joints(cell: Workcell, q: np.ndarray, duration: float = 1.0) -> State:
    """Ramp the 7 arm servo targets from the current command to q, then hold briefly."""
    ctrl = cell.data.ctrl.copy()
    q0 = ctrl[:7].copy()
    n = max(1, round(duration / (cell.model.opt.timestep * CHUNK)))
    for i in range(1, n + 1):
        s = i / n
        ctrl[:7] = q0 + (q - q0) * (3 * s**2 - 2 * s**3)  # smoothstep: zero start/end speed
        cell.step(ctrl, CHUNK)
    # The servos are overdamped and creep in after a long move: hold until every joint is
    # within SETTLE_RAD of its target (at least MIN_HOLD_S, at most MAX_HOLD_S if blocked).
    held, state = 0.0, cell.get_state()
    while held < MAX_HOLD_S:
        state = cell.step(ctrl, CHUNK)
        held += CHUNK * cell.model.opt.timestep
        if held >= MIN_HOLD_S and np.abs(np.asarray(state["qpos_arm"]) - q).max() < SETTLE_RAD:
            break
    return state


def move_tcp(cell: Workcell, pose: Pose, duration: float = 1.0) -> IKResult:
    """IK from the current command, then move there if IK converged. Returns the IK result."""
    ik = solve_ik(cell.model, cell.data, pose, q_init=cell.data.ctrl[:7].copy())
    if ik.converged:
        move_joints(cell, ik.q, duration)
    return ik


def set_gripper(cell: Workcell, open_: bool, duration: float = 0.5) -> State:
    ctrl = cell.data.ctrl.copy()
    ctrl[7] = mb.GRIPPER_OPEN_CTRL if open_ else mb.GRIPPER_CLOSED_CTRL
    return cell.step(ctrl, round(duration / cell.model.opt.timestep))


def heading(quat) -> float:
    """World-frame heading (radians) of a frame's x-axis: cube yaw, or TCP yaw."""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(quat, dtype=float))
    return float(np.arctan2(R[3], R[0]))


def grasp_quat(cube_quat, tcp_quat) -> tuple[float, float, float, float]:
    """Top-down TCP orientation for a cube, using its 90-degree symmetry to pick the yaw
    closest to the current hand yaw, so the wrist never spins more than 45 degrees."""
    now = heading(tcp_quat)
    d = heading(cube_quat) - now
    d = (d + np.pi / 4) % (np.pi / 2) - np.pi / 4
    return top_down_quat(now + d)

