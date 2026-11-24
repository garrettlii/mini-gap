"""Planning skills."""

from __future__ import annotations

import numpy as np

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
    description = "Release pose at a free spot in a zone, or on top of another cube when stacking."
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
            x, y, surface = _free_spot(s, zones[target]["pos"])
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


# Candidate release spots, meters from the zone center: the center, then the four corners
# (a 4 cm cube at a corner still lies fully inside the 14 cm zone).
ZONE_SLOTS = ((0.0, 0.0), (0.045, 0.045), (-0.045, -0.045), (0.045, -0.045), (-0.045, 0.045))


def _free_spot(state, zone_pos) -> tuple[float, float, float]:
    """The slot farthest from cubes already resting in the zone (held cubes don't count)."""
    resting = [p.pos for p in state["objects"].values() if p.pos[2] < mb.CUBE_HALF + 0.01]

    def clearance(slot):
        x, y = zone_pos[0] + slot[0], zone_pos[1] + slot[1]
        return min([np.hypot(x - c[0], y - c[1]) for c in resting] + [np.inf])

    dx, dy = max(ZONE_SLOTS, key=clearance)  # ties keep list order, so an empty zone -> center
    return zone_pos[0] + dx, zone_pos[1] + dy, zone_pos[2]
