"""Grasp skills."""

from __future__ import annotations

from gap.skills.base import Param, Skill
from gap.skills.registry import register
from gap.types import SkillResult


@register
class PlanTopGrasp(Skill):
    name = "plan_top_grasp"
    kind = "grasp"
    description = "Top-down grasp for a cube: pre-grasp pose above it and the grasp pose itself."
    inputs = {"pose": "Pose"}
    outputs = {"pre_pose": "Pose", "grasp_pose": "Pose"}
    params = {
        "approach_height": Param(0.08, 0.03, 0.20, "pre-grasp height above the grasp, meters"),
        "yaw_offset": Param(0.0, -1.5708, 1.5708, "added to the cube's yaw, radians"),
        "depth": Param(0.0, -0.015, 0.01, "grasp z offset from cube center, meters"),
    }

    def pre_margins(self, cell, inputs, params):
        # Hint: IK residual for the grasp pose and its distance from the base. Arbiter features.
        return {}

    def run(self, cell, inputs, params) -> SkillResult:
        # Hint: a cube is symmetric every 90 degrees; pick the yaw closest to the current hand
        # yaw so the wrist does not spin half a turn.
        raise NotImplementedError

    def post(self, cell, inputs, outputs, params):
        raise NotImplementedError


@register
class CloseGripper(Skill):
    name = "close_gripper"
    kind = "grasp"
    description = "Close the gripper and wait for the fingers to settle."
    params = {
        "force": Param(0.6, 0.1, 1.0, "fraction of full closing command"),
        "wait_steps": Param(150, 20, 500, "sim steps to wait"),
    }

    def run(self, cell, inputs, params) -> SkillResult:
        raise NotImplementedError

    def post(self, cell, inputs, outputs, params):
        # Width above ~0 (something is between the fingers) and both fingers touch the object.
        raise NotImplementedError


@register
class OpenGripper(Skill):
    name = "open_gripper"
    kind = "grasp"
    description = "Open the gripper fully."
    params = {"wait_steps": Param(100, 20, 500, "sim steps to wait")}

    def run(self, cell, inputs, params) -> SkillResult:
        raise NotImplementedError

    def post(self, cell, inputs, outputs, params):
        raise NotImplementedError


@register
class VerifyGrasp(Skill):
    name = "verify_grasp"
    kind = "check"
    description = "Lift slightly and confirm the object rises with the hand."
    inputs = {"color": "str"}
    params = {
        "lift": Param(0.03, 0.01, 0.08, "test lift, meters"),
        "width_min": Param(0.03, 0.0, 0.06, "finger width band, meters"),
        "width_max": Param(0.045, 0.02, 0.08, "finger width band, meters"),
    }

    def run(self, cell, inputs, params) -> SkillResult:
        raise NotImplementedError

    def post(self, cell, inputs, outputs, params):
        raise NotImplementedError
