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

from gap.types import Pose


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
    raise NotImplementedError


def top_down_quat(yaw: float) -> tuple[float, float, float, float]:
    """TCP orientation pointing straight down, rotated by yaw about world z.

    Hint: the hand's +z must point along world -z. Build it as a rotation of pi about
    world x, followed by yaw about world z; compose with mujoco.mju_mulQuat. Remember the
    Menagerie hand body has a built-in -45 degree offset about its own z (see panda.xml),
    so check the finger direction in the viewer before trusting your yaw.
    """
    raise NotImplementedError
