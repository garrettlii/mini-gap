"""Perception skills.

Interfaces (name, ports, params) are filled in from the build guide so the
validator and catalog work from day one. Implement run() and post().
"""

from __future__ import annotations

import numpy as np

from gap.sim import model_builder as mb
from gap.sim.motion import move_joints
from gap.sim.task_sampler import task_config
from gap.skills._common import cube_body
from gap.skills.base import Param, Skill
from gap.skills.registry import register
from gap.types import Pose, SkillResult

HUES = {"red": 0.0, "green": 1 / 3, "blue": 0.6}   # cube colors in workcell.xml, HSV hue
HUE_TOL = 0.06
# Arm swung 90 degrees to +y: the overhead camera then sees the whole spawn envelope.
CLEAR_VIEW_Q = np.array([1.5708, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7853])


def _plausible(pose: Pose) -> bool:
    """On or above the table, within the table's footprint. No ground truth used."""
    x, y, z = pose.pos
    return 0.0 <= x <= 0.95 and abs(y) <= 0.7 and -0.005 <= z <= 0.5


@register
class LocateObjectGT(Skill):
    name = "locate_object_gt"
    kind = "perception"
    description = "Ground-truth pose of the cube with the given color (privileged; for debugging)."
    inputs = {"color": "str"}
    outputs = {"pose": "Pose"}

    def run(self, cell, inputs, params) -> SkillResult:
        body = cube_body(inputs["color"])
        if body is None:
            return SkillResult(False, message=f"no cube with color {inputs['color']!r}")
        return SkillResult(True, {"pose": cell.get_state()["objects"][body]})

    def post(self, cell, inputs, outputs, params):
        ok = "pose" in outputs and _plausible(outputs["pose"])
        return ok, "" if ok else "pose missing or off the table"


@register
class LocateObjectRGBD(Skill):
    name = "locate_object_rgbd"
    kind = "perception"
    description = ("Estimate a colored cube's pose from the overhead RGB-D camera "
                   "(swings the arm aside for the picture, then back).")
    inputs = {"color": "str"}
    outputs = {"pose": "Pose"}
    params = {
        "sat_min": Param(0.5, 0.1, 0.9, "HSV saturation threshold"),
        "min_pixels": Param(30, 5, 500, "reject detections smaller than this"),
    }

    def run(self, cell, inputs, params) -> SkillResult:
        if inputs["color"] not in HUES:
            return SkillResult(False, message=f"no cube with color {inputs['color']!r}")
        q_back = cell.data.ctrl[:7].copy()
        move_joints(cell, CLEAR_VIEW_Q, duration=1.5)
        rgb, depth = cell.render("overhead"), cell.render("overhead", depth=True)
        move_joints(cell, q_back, duration=1.5)  # later motion should start from where it was
        mask = _color_mask(rgb, HUES[inputs["color"]], params["sat_min"])
        if mask.sum() < params["min_pixels"]:
            return SkillResult(False, message=f"{int(mask.sum())} {inputs['color']} pixels, "
                                              f"need {int(params['min_pixels'])}")
        pose = _mask_to_pose(cell, mask, depth)
        return SkillResult(True, {"pose": pose})

    def post(self, cell, inputs, outputs, params):
        # Plausibility only: at run time there is no ground truth to compare against.
        env = task_config()["envelope"]
        ok = "pose" in outputs and _plausible(outputs["pose"])
        ok = ok and env["x"][0] - 0.1 <= outputs["pose"].pos[0] <= env["x"][1] + 0.1
        return ok, "" if ok else "estimate is off the table or far outside the workspace"


def _color_mask(rgb: np.ndarray, hue: float, sat_min: float) -> np.ndarray:
    hsv = _rgb_to_hsv(rgb)
    dh = np.abs(hsv[..., 0] - hue)
    dh = np.minimum(dh, 1 - dh)  # hue wraps around at red
    return (dh < HUE_TOL) & (hsv[..., 1] >= sat_min) & (hsv[..., 2] > 0.2)


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Vectorized colorsys.rgb_to_hsv for an (H, W, 3) uint8 image; channels in [0, 1]."""
    x = rgb.astype(float) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx, mn = x.max(-1), x.min(-1)
    span = np.maximum(mx - mn, 1e-9)
    h = np.select([mx == r, mx == g], [(g - b) / span % 6, (b - r) / span + 2],
                  (r - g) / span + 4) / 6
    h = np.where(mx == mn, 0.0, h)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-9), 0.0)
    return np.stack([h, sat, mx], -1)


def _mask_to_pose(cell, mask: np.ndarray, depth: np.ndarray) -> Pose:
    """Back-project masked pixels through the overhead pinhole camera; yaw from the outline."""
    m = cell.model
    cam = m.camera("overhead").id
    h, w = mask.shape
    f = 0.5 * h / np.tan(np.deg2rad(m.cam_fovy[cam]) / 2)
    v, u = np.nonzero(mask)
    z = depth[v, u]
    # Camera frame: x right, y up, looking along -z.
    pts_cam = np.stack([(u - w / 2 + 0.5) * z / f, -(v - h / 2 + 0.5) * z / f, -z], -1)
    cam_pos, R = cell.calibrated_camera("overhead")   # nominal pose: calibration error stays
    pts = pts_cam @ R.T + cam_pos
    top = pts[pts[:, 2] > pts[:, 2].max() - 0.01]     # keep the top face only
    center = top.mean(0) - np.array([0, 0, mb.CUBE_HALF])
    yaw = _square_yaw(top[:, :2] - top[:, :2].mean(0))
    return Pose(tuple(center), (float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))))


def _square_yaw(xy: np.ndarray) -> float:
    """Yaw in [-pi/4, pi/4) whose axis-aligned bounding box around the points is smallest."""
    angles = np.deg2rad(np.arange(-45, 45, 1.0))
    areas = []
    for a in angles:
        c, s = np.cos(a), np.sin(a)
        r = xy @ np.array([[c, -s], [s, c]])  # rotate points by -a
        areas.append(np.ptp(r[:, 0]) * np.ptp(r[:, 1]))
    return float(angles[int(np.argmin(areas))])
