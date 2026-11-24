"""Motion skills."""

from __future__ import annotations

import numpy as np

from gap.sim import model_builder as mb
from gap.sim.ik import solve_ik
from gap.sim.motion import grasp_quat, move_joints
from gap.skills._common import GRIPPER_BODIES, move_duration, reachable
from gap.skills.base import Param, Skill
from gap.skills.registry import register
from gap.types import Pose, SkillResult


def _go(cell, pose: Pose, speed: float) -> SkillResult:
    """IK + ramped move. outputs["target"] is internal: the pose post() checks against."""
    ik = solve_ik(cell.model, cell.data, pose, q_init=cell.data.ctrl[:7].copy())
    if not ik.converged:
        return SkillResult(False, message=f"pose {pose.pos} unreachable "
                                          f"(IK residual {ik.pos_err * 100:.1f} cm)")
    move_joints(cell, ik.q, move_duration(cell, ik.q, speed))
    return SkillResult(True, {"target": pose})


def _arrived(cell, pose: Pose, tol: float) -> tuple[bool, str]:
    err = float(np.linalg.norm(cell.get_state()["tcp"].p - pose.p))
    return (True, "") if err <= tol else (False, f"TCP stopped {err * 100:.1f} cm from target")


def _above(cell, pose: Pose, height: float) -> Pose:
    """Point straight down over pose, at the cube-symmetric yaw nearest the hand's yaw."""
    q = grasp_quat(pose.quat, cell.get_state()["tcp"].quat)
    return Pose((pose.pos[0], pose.pos[1], pose.pos[2] + height), q)


@register
class MoveToPose(Skill):
    name = "move_to_pose"
    kind = "motion"
    description = "Move the tool center point to a target pose along an interpolated joint path."
    inputs = {"pose": "Pose"}
    params = {
        "speed": Param(0.5, 0.1, 1.0, "fraction of max joint speed"),
        "tolerance": Param(0.005, 0.001, 0.02, "meters"),
    }

    def pre_margins(self, cell, inputs, params):
        _, residual = reachable(cell, inputs["pose"])
        return {"ik_residual": residual, "reach": float(np.hypot(*inputs["pose"].pos[:2]))}

    def run(self, cell, inputs, params) -> SkillResult:
        return _go(cell, inputs["pose"], params["speed"])

    def post(self, cell, inputs, outputs, params):
        return _arrived(cell, inputs["pose"], params["tolerance"])


@register
class MoveAbove(Skill):
    name = "move_above"
    kind = "motion"
    description = "Move the tool center point to a height above a pose, pointing down."
    inputs = {"pose": "Pose"}
    params = {"height": Param(0.10, 0.03, 0.30, "meters above the pose")}

    def run(self, cell, inputs, params) -> SkillResult:
        return _go(cell, _above(cell, inputs["pose"], params["height"]), speed=0.5)

    def post(self, cell, inputs, outputs, params):
        return _arrived(cell, outputs["target"], 0.01)


@register
class Retreat(Skill):
    name = "retreat"
    kind = "motion"
    description = "Lift straight up, clear of all objects."
    params = {"height": Param(0.15, 0.05, 0.30, "meters")}

    def run(self, cell, inputs, params) -> SkillResult:
        return _go(cell, cell.get_state()["tcp"].offset(dz=params["height"]), speed=0.5)

    def post(self, cell, inputs, outputs, params):
        ok, msg = _arrived(cell, outputs["target"], 0.01)
        s = cell.get_state()
        hits = [p for p in s["contacts"] if set(p) & set(GRIPPER_BODIES) and set(p) & set(mb.CUBES)]
        if hits:
            return False, f"gripper still touching {hits}"
        return ok, msg
