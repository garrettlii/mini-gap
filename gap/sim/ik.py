"""Inverse kinematics for the 7-DoF arm.

Write this yourself first: it is the most transferable piece of robotics math in
the project. Compare with mink (github.com/kevinzakka/mink) only after yours works.

Damped least squares, one iteration:
    e   = [p_target - p_tcp ; orientation_error]       (6,)
    J   = [jacp ; jacr] restricted to the 7 arm dofs   (6, 7)
    dq  = J^T (J J^T + lambda^2 I)^-1 e
    q  <- clip(q + step * dq, joint limits)

Hints:
  - mujoco.mj_jacSite(model, data, jacp, jacr, site_id) fills (3, nv) arrays.
  - Arm dof columns: model.jnt_dofadr[model.joint(name).id] for each name in ARM_JOINTS.
  - Orientation error: mujoco.mju_subQuat(err3, q_target, q_current) gives a 3-vector.
  - Iterate on a scratch MjData (mujoco.MjData(model), copy qpos in) and call
    mujoco.mj_kinematics + mujoco.mj_comPos each iteration, so the live sim is untouched.
  - Joint limits: model.jnt_range. Treat orientation as lower priority than position by
    weighting the rotation rows (for example 0.3).
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from gap.sim import model_builder as mb
from gap.types import Pose

WRIST_SEEDS = (-2.5, -1.25, 0.0, 1.25, 2.5)  # joint 7 restart values, radians


@dataclass
class IKResult:
    q: np.ndarray          # (7,) arm joint targets
    pos_err: float         # meters
    rot_err: float         # radians
    iters: int
    converged: bool


def solve_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target: Pose,
    *,
    q_init: np.ndarray | None = None,
    damping: float = 1e-2,
    step: float = 0.5,
    max_iters: int = 200,
    pos_tol: float = 2e-3,
    rot_tol: float = 2e-2,
    rot_weight: float = 0.3,
) -> IKResult:
    m = model
    scratch = mujoco.MjData(m)  # never touch the live sim
    scratch.qpos[:] = data.qpos
    site = m.site(mb.TCP_SITE).id
    jids = [m.joint(n).id for n in mb.ARM_JOINTS]
    qadr, dadr = m.jnt_qposadr[jids], m.jnt_dofadr[jids]
    lo, hi = m.jnt_range[jids].T
    q0 = scratch.qpos[qadr].copy() if q_init is None else np.asarray(q_init, dtype=float)
    # Restarts: the wrist turns less than a full circle, so the short way round can hit its
    # limit when the target lies the long way round. Re-seed joint 7 across its range.
    starts = [q0] + [np.r_[q0[:6], q7] for q7 in WRIST_SEEDS]
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    w = np.array([1.0] * 3 + [rot_weight] * 3)
    best: IKResult | None = None
    for q_start in starts:
        scratch.qpos[qadr] = q_start
        for it in range(1, max_iters + 1):
            pos_e, rot_e = _tcp_error(m, scratch, site, target)
            pos_err, rot_err = float(np.linalg.norm(pos_e)), float(np.linalg.norm(rot_e))
            if pos_err < pos_tol and rot_err < rot_tol:
                return IKResult(scratch.qpos[qadr].copy(), pos_err, rot_err, it, True)
            mujoco.mj_jacSite(m, scratch, jacp, jacr, site)
            J = np.vstack([jacp, jacr])[:, dadr] * w[:, None]
            e = np.concatenate([pos_e, rot_e]) * w
            q = scratch.qpos[qadr]
            dq = _dls(J, e, damping)
            # A joint pinned at a limit can't follow dq; drop it so the others take the slack.
            blocked = ((q <= lo + 1e-6) & (dq < 0)) | ((q >= hi - 1e-6) & (dq > 0))
            if blocked.any():
                J[:, blocked] = 0.0
                dq = _dls(J, e, damping)
            scratch.qpos[qadr] = np.clip(q + step * dq, lo, hi)
        r = IKResult(scratch.qpos[qadr].copy(), pos_err, rot_err, max_iters, False)
        if best is None or r.pos_err + r.rot_err < best.pos_err + best.rot_err:
            best = r
    assert best is not None
    return best


def _tcp_error(m: mujoco.MjModel, d: mujoco.MjData, site: int, target: Pose):
    """(position error, rotation error), both in the world frame like mj_jacSite."""
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)  # mj_jacSite needs com-based quantities
    cur, err = np.zeros(4), np.zeros(3)
    mujoco.mju_mat2Quat(cur, d.site_xmat[site])
    mujoco.mju_subQuat(err, target.q, cur)  # expressed in the TCP frame
    return target.p - d.site_xpos[site], d.site_xmat[site].reshape(3, 3) @ err


def top_down_quat(yaw: float) -> tuple[float, float, float, float]:
    """TCP orientation pointing straight down, rotated by yaw about world z.

    Hint: the hand's +z must point along world -z. Build it as a rotation of pi about
    world x, followed by yaw about world z; compose with mujoco.mju_mulQuat. Remember the
    Menagerie hand body has a built-in -45 degree offset about its own z (see panda.xml),
    so check the finger direction in the viewer before trusting your yaw.

    yaw is the heading of the TCP x-axis; the fingers close along yaw - pi/2.
    Home pose is yaw = pi/2. Closed form of quat_z(yaw) * quat_x(pi).
    """
    return (0.0, float(np.cos(yaw / 2)), float(np.sin(yaw / 2)), 0.0)


def _dls(J: np.ndarray, e: np.ndarray, damping: float) -> np.ndarray:
    """Damped least squares step: J^T (J J^T + lambda^2 I)^-1 e."""
    return J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(J.shape[0]), e)
