"""Planning skills."""

from __future__ import annotations

from gap.sim import model_builder as mb
from gap.sim.ik import top_down_quat
from gap.sim.motion import heading
from gap.sim.task_sampler import task_config
from gap.skills._common import cube_body, reachable
from gap.skills.base import Param, Skill
from gap.skills.registry import register
from gap.types import Pose, SkillResult


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
        s = cell.get_state()
        target = inputs["target"]
        zones = task_config()["zones"]
        if target in zones:
            x, y, surface = zones[target]["pos"]
        elif cube_body(target):
            x, y, z = s["objects"][cube_body(target)].pos
            surface = z + mb.CUBE_HALF + params["stack_offset"]
        else:
            return SkillResult(False, message=f"target {target!r} is neither a zone "
                                              f"({', '.join(zones)}) nor a cube color")
        # TCP sits at the held cube's center: half a cube above the surface, plus the drop.
        q = top_down_quat(heading(s["tcp"].quat))  # keep the hand's current yaw
        return SkillResult(True, {"pose": Pose((x, y, surface + mb.CUBE_HALF + params["drop_height"]), q)})

    def post(self, cell, inputs, outputs, params):
        ok = reachable(cell, outputs["pose"])[0]
        return (True, "") if ok else (False, f"place pose {outputs['pose'].pos} is out of reach")
