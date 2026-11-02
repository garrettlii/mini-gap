"""Perception skills.

Interfaces (name, ports, params) are filled in from the build guide so the
validator and catalog work from day one. Implement run() and post().
"""

from __future__ import annotations

from gap.skills.base import Param, Skill
from gap.skills.registry import register
from gap.types import SkillResult


@register
class LocateObjectGT(Skill):
    name = "locate_object_gt"
    kind = "perception"
    description = "Ground-truth pose of the cube with the given color (privileged; for debugging)."
    inputs = {"color": "str"}
    outputs = {"pose": "Pose"}

    def run(self, cell, inputs, params) -> SkillResult:
        raise NotImplementedError

    def post(self, cell, inputs, outputs, params):
        raise NotImplementedError


@register
class LocateObjectRGBD(Skill):
    name = "locate_object_rgbd"
    kind = "perception"
    description = "Estimate a colored cube's pose from the overhead RGB-D camera."
    inputs = {"color": "str"}
    outputs = {"pose": "Pose"}
    params = {
        "sat_min": Param(0.5, 0.1, 0.9, "HSV saturation threshold"),
        "min_pixels": Param(30, 5, 500, "reject detections smaller than this"),
    }

    def pre(self, cell, inputs, params):
        # Hint: the arm blocks the overhead camera in the home pose. Check the cube is not
        # occluded, or require a 'clear view' arm pose before this node.
        return True, ""

    def run(self, cell, inputs, params) -> SkillResult:
        # Hint: HSV mask -> pixel centroid -> depth -> back-project with the camera's fovy and
        # pose (model.cam_fovy, data.cam_xpos, data.cam_xmat). Yaw from the mask's principal axis.
        raise NotImplementedError

    def post(self, cell, inputs, outputs, params):
        # Tests may compare against ground truth (within 1 cm). post() itself must not:
        # at run time it can only check plausibility (on the table, inside the envelope).
        raise NotImplementedError
