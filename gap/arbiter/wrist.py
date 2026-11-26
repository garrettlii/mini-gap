"""What the learned grasp sees through the wrist camera (DECISIONS 2026-11-26).

The privileged observation (fallback.observe) reads the true hand-object offsets. This one
has to find the object: render the wrist RGB-D camera, segment each cube color, back-project
through the calibrated hand-eye pose, keep the top face, and take the blob nearest the TCP.
The camera sits 6 cm off the finger axis, so near the object the top face can leave the
image or hide behind a finger, and the estimate degrades in ways the scripted skill (which
trusts the overhead estimate instead) does not share.

Vector layout: the privileged one plus `seen`, the visible top-face area over a full square
face of the estimated size (0 when nothing is found; then every offset is 0 as well).
"""

from __future__ import annotations

import numpy as np

from gap.sim.motion import heading
from gap.skills.perception import HUES, _color_mask, _square_yaw, back_project, focal_px

SAT_MIN = 0.5          # the RGB-D locator's default saturation threshold
MIN_PIXELS = 15        # fewer top-face pixels than this: no detection
TOP_BAND = 0.005       # meters below the highest point still counted as the top face
ROUND_FILL = 0.9       # top-face fill of its bounding square: square ~1, disc ~0.79
MIN_TOP_Z = 0.01       # an object stands above the table; the cyan zone and blue floor do not


def observe_wrist(cell, rng: np.random.Generator) -> np.ndarray:
    """[close_axis, width_axis, dz, yaw, half, round, gripper width, tcp height, seen]."""
    rgb, depth = cell.render("wrist"), cell.render("wrist", depth=True)
    state = cell.get_state()
    tcp = state["tcp"]
    blobs = [b for hue in HUES.values() if (b := _top_face(cell, rgb, depth, hue)) is not None]
    tail = [state["gripper_width"], tcp.pos[2]]
    if not blobs:
        return np.r_[np.zeros(6), tail, 0.0]
    top, px_area = min(blobs, key=lambda b: np.hypot(*(b[0][:, :2].mean(0) - tcp.pos[:2])))
    return np.r_[_offsets(top, px_area, tcp), tail, _seen(top, px_area)]


def _top_face(cell, rgb, depth, hue) -> tuple[np.ndarray, float] | None:
    """(top-face world points, world area per pixel) of one color, or None if too small."""
    mask = _color_mask(rgb, hue, SAT_MIN)
    if mask.sum() < MIN_PIXELS:
        return None
    pts = back_project(cell, "wrist", mask, depth)
    top = pts[pts[:, 2] > pts[:, 2].max() - TOP_BAND]
    if len(top) < MIN_PIXELS or top[:, 2].mean() < MIN_TOP_Z:
        return None
    cam_z = cell.calibrated_camera("wrist")[0][2]
    px = (cam_z - top[:, 2].mean()) / focal_px(cell, "wrist", mask.shape[0])
    return top, float(px**2)


def _offsets(top: np.ndarray, px_area: float, tcp) -> np.ndarray:
    """Same quantities and frame as skills._common.hand_offsets, from the top-face points."""
    xy, top_z = top[:, :2].mean(0), float(top[:, 2].mean())
    half = top_z / 2                                  # resting on the table (z = 0)
    hand_yaw = heading(tcp.quat)
    close_dir = hand_yaw - np.pi / 2                  # fingers close along yaw - pi/2
    dx, dy = xy - tcp.pos[:2]
    obj_yaw = _square_yaw(top[:, :2] - xy)
    round_ = float(_fill(top, px_area, obj_yaw) < ROUND_FILL)
    dyaw = (obj_yaw - hand_yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
    return np.array([dx * np.cos(close_dir) + dy * np.sin(close_dir),
                     -dx * np.sin(close_dir) + dy * np.cos(close_dir),
                     half - tcp.pos[2], 0.0 if round_ else dyaw, half, round_])


def _fill(top: np.ndarray, px_area: float, yaw: float) -> float:
    c, s = np.cos(yaw), np.sin(yaw)
    r = (top[:, :2] - top[:, :2].mean(0)) @ np.array([[c, -s], [s, c]])
    return len(top) * px_area / max(np.ptp(r[:, 0]) * np.ptp(r[:, 1]), 1e-9)


def _seen(top: np.ndarray, px_area: float) -> float:
    side = float(top[:, 2].mean())                    # top height = 2 x half = face side
    return len(top) * px_area / max(side**2, 1e-9)
