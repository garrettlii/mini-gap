"""Grasp skills."""

from __future__ import annotations

import numpy as np

from gap.sim import model_builder as mb
from gap.sim.ik import top_down_quat
from gap.sim.motion import grasp_quat, heading, move_tcp
from gap.skills._common import cube_body, reachable, touching
from gap.skills.base import Param, Skill
from gap.skills.registry import register
from gap.types import Pose, SkillResult

MAX_WIDTH = 0.08  # meters, fully open
CUBE_WIDTH_CTRL = mb.GRIPPER_OPEN_CTRL * 2 * mb.CUBE_HALF / MAX_WIDTH


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
        grasp = self._poses(cell, inputs["pose"], params)[1]
        _, residual = reachable(cell, grasp)
        return {"ik_residual": residual, "reach": float(np.hypot(*grasp.pos[:2]))}

    def run(self, cell, inputs, params) -> SkillResult:
        pre, grasp = self._poses(cell, inputs["pose"], params)
        return SkillResult(True, {"pre_pose": pre, "grasp_pose": grasp})

    def post(self, cell, inputs, outputs, params):
        for port in ("pre_pose", "grasp_pose"):
            if not reachable(cell, outputs[port])[0]:
                return False, f"{port} {outputs[port].pos} is out of reach (IK did not converge)"
        return True, ""

    @staticmethod
    def _poses(cell, cube: Pose, params) -> tuple[Pose, Pose]:
        # A cube repeats every 90 degrees: grasp_quat picks the yaw nearest the hand's yaw.
        q = grasp_quat(cube.quat, cell.get_state()["tcp"].quat)
        q = top_down_quat(heading(q) + params["yaw_offset"])
        grasp = Pose((cube.pos[0], cube.pos[1], cube.pos[2] + params["depth"]), q)
        return grasp.offset(dz=params["approach_height"]), grasp


@register
class CloseGripper(Skill):
    name = "close_gripper"
    kind = "grasp"
    description = "Close the gripper and wait for the fingers to settle."
    params = {
        "force": Param(0.6, 0.1, 1.0, "fraction of max squeeze; 1.0 = full closing command"),
        "wait_steps": Param(150, 20, 500, "sim steps to wait"),
    }

    def run(self, cell, inputs, params) -> SkillResult:
        ctrl = cell.data.ctrl.copy()
        # Squeeze grows with how far the command is below the cube's width, so scale from a
        # cube-width opening (no squeeze) to fully closed (max squeeze).
        ctrl[7] = CUBE_WIDTH_CTRL * (1 - params["force"])
        n = int(params["wait_steps"])
        cell.step(ctrl, n)
        return SkillResult(True, sim_steps=n)

    def post(self, cell, inputs, outputs, params):
        s = cell.get_state()
        fingers = {"left_finger", "right_finger"}
        touched = {c: touching(s, c, fingers) for c in mb.CUBES}
        if any(t == fingers for t in touched.values()):
            return True, ""
        partial = [f"{c} ({', '.join(sorted(t))})" for c, t in touched.items() if t]
        if partial:
            return False, f"only one finger touches {partial[0]}: grasp is off center"
        return False, f"fingers closed on nothing (width {s['gripper_width']:.3f} m)"


@register
class OpenGripper(Skill):
    name = "open_gripper"
    kind = "grasp"
    description = "Open the gripper fully."
    params = {"wait_steps": Param(100, 20, 500, "sim steps to wait")}

    def run(self, cell, inputs, params) -> SkillResult:
        ctrl = cell.data.ctrl.copy()
        ctrl[7] = mb.GRIPPER_OPEN_CTRL
        n = int(params["wait_steps"])
        cell.step(ctrl, n)
        return SkillResult(True, sim_steps=n)

    def post(self, cell, inputs, outputs, params):
        w = cell.get_state()["gripper_width"]
        return (True, "") if w > 0.06 else (False, f"gripper only opened to {w:.3f} m")


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
        if cube_body(inputs["color"]) is None:
            return SkillResult(False, message=f"no cube with color {inputs['color']!r}")
        tcp = cell.get_state()["tcp"]
        ik = move_tcp(cell, tcp.offset(dz=params["lift"]), duration=0.5)
        return SkillResult(ik.converged, message="" if ik.converged else "test lift unreachable")

    def post(self, cell, inputs, outputs, params):
        s = cell.get_state()
        body = cube_body(inputs["color"])
        lifted = s["objects"][body].pos[2] - mb.CUBE_HALF
        w = s["gripper_width"]
        if lifted < 0.5 * params["lift"]:
            return False, f"{body} rose only {lifted * 100:.1f} cm with the hand"
        if not params["width_min"] <= w <= params["width_max"]:
            return False, f"finger width {w:.3f} m outside [{params['width_min']}, {params['width_max']}]"
        return True, ""
