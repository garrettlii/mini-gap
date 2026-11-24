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
VIDEO_FPS = 25      # one frame every 2 chunks
SETTLE_S = 0.2      # hold at the goal so the servos catch up


def move_joints(cell: Workcell, q: np.ndarray, duration: float = 1.0,
                frame_sink: list | None = None) -> State:
    """Ramp the 7 arm servo targets from the current command to q, then hold briefly."""
    ctrl = cell.data.ctrl.copy()
    q0 = ctrl[:7].copy()
    n = max(1, round(duration / (cell.model.opt.timestep * CHUNK)))
    for i in range(1, n + 1):
        s = i / n
        ctrl[:7] = q0 + (q - q0) * (3 * s**2 - 2 * s**3)  # smoothstep: zero start/end speed
        _advance(cell, ctrl, CHUNK, frame_sink)
    return _advance(cell, ctrl, round(SETTLE_S / cell.model.opt.timestep), frame_sink)


def move_tcp(cell: Workcell, pose: Pose, duration: float = 1.0,
             frame_sink: list | None = None) -> IKResult:
    """IK from the current command, then move there if IK converged. Returns the IK result."""
    ik = solve_ik(cell.model, cell.data, pose, q_init=cell.data.ctrl[:7].copy())
    if ik.converged:
        move_joints(cell, ik.q, duration, frame_sink)
    return ik


def set_gripper(cell: Workcell, open_: bool, duration: float = 0.5,
                frame_sink: list | None = None) -> State:
    ctrl = cell.data.ctrl.copy()
    ctrl[7] = mb.GRIPPER_OPEN_CTRL if open_ else mb.GRIPPER_CLOSED_CTRL
    return _advance(cell, ctrl, round(duration / cell.model.opt.timestep), frame_sink)


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


def _advance(cell: Workcell, ctrl: np.ndarray, steps: int, frame_sink: list | None) -> State:
    if frame_sink is None:
        return cell.step(ctrl, steps)
    dt = cell.model.opt.timestep
    per_frame = round(1 / (VIDEO_FPS * dt))
    state = cell.step(ctrl, 0)
    for _ in range(steps):
        state = cell.step(None, 1)
        if round(cell.data.time / dt) % per_frame == 0:
            frame_sink.append(cell.render("front"))
    return state
