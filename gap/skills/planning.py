"""Planning skills."""

from __future__ import annotations

from gap.skills.base import Param, Skill
from gap.skills.registry import register
from gap.types import SkillResult


@register
class ComputePlacePose(Skill):
    name = "compute_place_pose"
    kind = "planning"
    description = "Release pose above a zone, or on top of another cube when stacking."
    inputs = {"target": "str"}          # a zone name ("zone_a") or a cube color ("red")
    outputs = {"pose": "Pose"}
    params = {
        "drop_height": Param(0.01, 0.0, 0.05, "release height above the surface, meters"),
        "stack_offset": Param(0.0, -0.01, 0.01, "extra z when stacking, meters"),
    }

    def run(self, cell, inputs, params) -> SkillResult:
        raise NotImplementedError

    def post(self, cell, inputs, outputs, params):
        raise NotImplementedError
