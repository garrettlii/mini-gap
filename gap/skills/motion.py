"""Motion skills."""

from __future__ import annotations

from gap.skills.base import Param, Skill
from gap.skills.registry import register
from gap.types import SkillResult


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

    def run(self, cell, inputs, params) -> SkillResult:
        raise NotImplementedError

    def post(self, cell, inputs, outputs, params):
        raise NotImplementedError


@register
class MoveAbove(Skill):
    name = "move_above"
    kind = "motion"
    description = "Move the tool center point to a height above a pose, pointing down."
    inputs = {"pose": "Pose"}
    params = {"height": Param(0.10, 0.03, 0.30, "meters above the pose")}

    def run(self, cell, inputs, params) -> SkillResult:
        raise NotImplementedError

    def post(self, cell, inputs, outputs, params):
        raise NotImplementedError


@register
class Retreat(Skill):
    name = "retreat"
    kind = "motion"
    description = "Lift straight up, clear of all objects."
    params = {"height": Param(0.15, 0.05, 0.30, "meters")}

    def run(self, cell, inputs, params) -> SkillResult:
        raise NotImplementedError

    def post(self, cell, inputs, outputs, params):
        raise NotImplementedError
