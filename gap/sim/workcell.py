"""Layer 1: the workcell.

Everything above this file talks to MuJoCo only through Workcell. Keep the
public surface to the methods below; add private helpers as you like.

Phase 1 build order (each step has a test in tests/test_phase1_workcell.py):
  1. get_state()            read poses, gripper width, contacts out of MjData
  2. reset(instance)        home keyframe + object poses from the instance
  3. step(ctrl, n)          write ctrl, advance n substeps
  4. snapshot() / restore() exact state copy, needed for retries and Phase 6b
  5. render(camera)         offscreen RGB for perception and videos
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from gap.sim import model_builder as mb
from gap.types import Pose, State, TaskInstance

SETTLE_STEPS = 100   # 0.2 s at dt = 2 ms: soft contacts come to rest
VIDEO_FPS = 25       # frames per sim second when frame_sink is set
SETTLE_TOL = 1e-3    # meters; a cube moving more than this while settling spawned badly
# Overhead RGB-D noise (camera_noise=True), see DECISIONS 2026-11-24 Phase 6b.
CALIB_ROT_SD = np.deg2rad(0.5)    # extrinsic rotation error per axis, per episode
CALIB_POS_SD = 0.005              # extrinsic translation error per axis, meters
DEPTH_SD_COEF = 2.85e-3           # depth noise sigma = coef * z^2 (Kinect-style)
# Wrist camera hand-eye calibration error, see DECISIONS 2026-11-26.
WRIST_ROT_SD = np.deg2rad(0.5)
WRIST_POS_SD = 0.002
NOISY_CAMERAS = ("overhead", "wrist")


class Workcell:
    def __init__(self, render_size: tuple[int, int] = (240, 320), camera_noise: bool = False,
                 video_size: tuple[int, int] | None = None):
        self.model: mujoco.MjModel = mb.build_model()
        self.camera_noise = camera_noise
        self._cam_nominal = {c: (self.model.camera(c).pos.copy(), self.model.camera(c).quat.copy())
                             for c in mb.CAMERAS}
        self._noise_rng = np.random.default_rng(0)
        self._wrist_rng = np.random.default_rng(0)   # wrist noise: own stream, overhead unchanged
        # geom ids per cube, indexed like mb.VARIANT_NAMES; the visible group of the main geom
        self._variant_geoms = {c: [self.model.geom(f"{c}_geom").id] +
                               [self.model.geom(f"{c}_{v}").id for v in mb.VARIANT_NAMES[1:]]
                               for c in mb.CUBES}
        self._shown_group = int(self.model.geom_group[self._variant_geoms[mb.CUBES[0]][0]])
        self.data: mujoco.MjData = mujoco.MjData(self.model)
        self.render_size = render_size          # (height, width)
        self._renderer: mujoco.Renderer | None = None  # create lazily: rehearsal workers rarely render
        # video frames get their own size so recording never changes what perception sees
        self.video_size = video_size or render_size
        self._video_renderer: mujoco.Renderer | None = None
        self.instance: TaskInstance | None = None
        self.frame_sink: list | None = None     # set to a list to record VIDEO_FPS front-camera frames

    # ------------------------------------------------------------------ state
    def get_state(self) -> State:
        """Return the State dict documented in gap/types.py.

        Hints:
          - Body pose: self.data.body(name).xpos and .xquat
          - Site pose: self.data.site(mb.TCP_SITE).xpos; orientation via .xmat (3x3, row-major,
            9 floats) -> mujoco.mju_mat2Quat
          - Gripper width: sum of the two finger joint positions (self.data.joint(name).qpos)
          - Contacts: loop over self.data.contact[:self.data.ncon]; each has geom1/geom2;
            map geom -> body with self.model.geom_bodyid, body id -> name with
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
        Return plain Python types (Pose, float, list), not numpy views into MjData:
        views change under you when the sim steps.
        """
        state = {}

        state["time"] = float(self.data.time)

        site = self.data.site(mb.TCP_SITE)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, site.xmat)  
        state["tcp"] = Pose.from_arrays(site.xpos, quat)

        left = self.data.joint(mb.FINGER_JOINTS[0]).qpos
        right = self.data.joint(mb.FINGER_JOINTS[1]).qpos
        state["gripper_width"] = float(left[0] + right[0])

        state["objects"] = {}
        for name in mb.CUBES:
            body = self.data.body(name)
            state["objects"][name] = Pose.from_arrays(body.xpos, body.xquat)

        contacts = set()
        for contact in self.data.contact[:self.data.ncon]:
            geom1 = contact.geom1
            geom2 = contact.geom2
            body1_id = self.model.geom_bodyid[geom1]
            body2_id = self.model.geom_bodyid[geom2]
            body1_name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                int(body1_id)
            )
            body2_name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                int(body2_id)
            )
            
            contacts.add(tuple(sorted([body1_name, body2_name])))
        state["contacts"] = sorted(contacts)
            
        state["qpos_arm"] = []
        for joint_name in mb.ARM_JOINTS:
            angle = self.data.joint(joint_name).qpos[0]
            state["qpos_arm"].append(float(angle))

        return state

    
    def reset(self, instance: TaskInstance) -> State:
        """Reset to the home keyframe, then place each object at instance.object_poses.

        Hints:
          - mujoco.mj_resetDataKeyframe(model, data, model.key(mb.HOME_KEY).id)
          - A freejoint's qpos is 7 numbers [x y z qw qx qy qz]; find its start index with
            model.jnt_qposadr[model.joint("red_cube_joint").id]. Zero its qvel too.
          - Call mujoco.mj_forward afterwards so xpos is up to date.
          - Let objects settle for a few dozen steps and check none moved more than 1 mm;
            a cube that jumps means it spawned interpenetrating something.
        """
        m, d = self.model, self.data
        self.instance = instance
        for cube in mb.CUBES:
            self._show_variant(cube, int(instance.features.get(f"{cube}_variant", 0)))
        self._set_camera_error(instance.seed)
        mujoco.mj_resetDataKeyframe(m, d, m.key(mb.HOME_KEY).id)
        for name, pose in instance.object_poses.items():
            joint = d.joint(f"{name}_joint")  # named accessor slices qpos by qposadr, qvel by dofadr
            joint.qpos[:] = [*pose.pos, *pose.quat]
            joint.qvel[:] = 0.0
        mujoco.mj_forward(m, d)  # xpos is stale until forward kinematics runs

        start = {name: d.body(name).xpos.copy() for name in instance.object_poses}
        mujoco.mj_step(m, d, SETTLE_STEPS)
        for name, p0 in start.items():
            moved = float(np.linalg.norm(d.body(name).xpos - p0))
            if moved > SETTLE_TOL:
                raise RuntimeError(f"{name} moved {moved * 1e3:.1f} mm while settling "
                                   f"(seed {instance.seed}): it spawned overlapping something")
        return self.get_state()

    def _show_variant(self, cube: str, variant: int) -> None:
        """Only the chosen variant geom collides and renders; the others are inert."""
        for i, gid in enumerate(self._variant_geoms[cube]):
            on = i == variant
            self.model.geom_contype[gid] = 1 if on else 2      # 2: in the BVH, collides with nothing
            self.model.geom_conaffinity[gid] = int(on)
            self.model.geom_group[gid] = self._shown_group if on else mb.HIDDEN_GROUP

    def _set_camera_error(self, seed: int) -> None:
        """Perturb the overhead and wrist cameras' true poses (calibration error); perception
        keeps using the nominal poses from calibrated_camera()."""
        self._noise_rng = np.random.default_rng([seed, 7])
        self._wrist_rng = np.random.default_rng([seed, 8])
        for name, rng, rot_sd, pos_sd in (("overhead", self._noise_rng, CALIB_ROT_SD, CALIB_POS_SD),
                                          ("wrist", self._wrist_rng, WRIST_ROT_SD, WRIST_POS_SD)):
            cam = self.model.camera(name).id
            pos, quat = self._cam_nominal[name]
            if not self.camera_noise:
                self.model.cam_pos[cam], self.model.cam_quat[cam] = pos, quat
                continue
            r = rng.normal(0.0, rot_sd, 3)
            dq, out = np.zeros(4), np.zeros(4)
            mujoco.mju_axisAngle2Quat(dq, r / max(np.linalg.norm(r), 1e-12), np.linalg.norm(r))
            mujoco.mju_mulQuat(out, dq, quat)
            self.model.cam_pos[cam] = pos + rng.normal(0.0, pos_sd, 3)
            self.model.cam_quat[cam] = out

    def calibrated_camera(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """(position, 3x3 rotation) in the world of a camera as calibrated: its nominal pose
        on its parent body, even when the true pose carries calibration error."""
        pos, quat = self._cam_nominal[name]
        body = self.model.cam_bodyid[self.model.camera(name).id]
        R_body = self.data.xmat[body].reshape(3, 3)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, quat)
        return self.data.xpos[body] + R_body @ pos, R_body @ R.reshape(3, 3)

    def step(self, ctrl: np.ndarray | None = None, n: int = 1) -> State:
        """Write ctrl (length model.nu: 7 arm joint targets + gripper 0..255) and step n times.

        ctrl=None keeps the previous control. Return get_state() after stepping.
        """

        if ctrl is not None:
            ctrl = np.asarray(ctrl, dtype=float)
            if ctrl.shape != (self.model.nu,):
                raise ValueError(f"ctrl shape {ctrl.shape}, expected ({self.model.nu},)")
            # MuJoCo clamps internally but stores the raw value; clip so logged ctrl == applied ctrl.
            lo, hi = self.model.actuator_ctrlrange.T
            self.data.ctrl[:] = np.clip(ctrl, lo, hi)
        dt = self.model.opt.timestep
        per_frame = round(1 / (VIDEO_FPS * dt))
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
            if self.frame_sink is not None and round(self.data.time / dt) % per_frame == 0:
                self.frame_sink.append(self._video_frame())
        return self.get_state()

    # --------------------------------------------------------- snapshot/restore
    def snapshot(self) -> Any:
        """Exact copy of the physics state (qpos, qvel, act, ctrl, time, warmstart).

        Hint: mujoco.mj_stateSize / mujoco.mj_getState with
        mujoco.mjtState.mjSTATE_INTEGRATION. Test: snapshot, step 200, restore, step 200
        again; both rollouts must match bit for bit.
        """
        size = mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_INTEGRATION)
        snap = np.empty(size, dtype=np.float64)
        mujoco.mj_getState(self.model, self.data, snap, mujoco.mjtState.mjSTATE_INTEGRATION)
        return snap

    def restore(self, snap: Any) -> State:
        """Inverse of snapshot(). Remember mj_forward after mj_setState."""
        mujoco.mj_setState(self.model, self.data, snap, mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(self.model, self.data)
        return self.get_state()

    # ---------------------------------------------------------------- render
    def render(self, camera: str = "front", depth: bool = False) -> np.ndarray:
        """RGB uint8 image (H, W, 3), or float depth in meters if depth=True.

        Hints: mujoco.Renderer(model, height, width); update_scene(data, camera=camera);
        enable_depth_rendering() / disable_depth_rendering() to switch modes.
        Cameras: mb.CAMERAS. Note the overhead view is blocked by the arm in the home pose.
        """
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height = self.render_size[0], width = self.render_size[1])

        self._renderer.update_scene(self.data, camera=camera)
        if depth:
            self._renderer.enable_depth_rendering()
            try:
                image = self._renderer.render()
            finally:  # depth mode is sticky; never leave it on after a failed render
                self._renderer.disable_depth_rendering()
            if self.camera_noise and camera in NOISY_CAMERAS:
                rng = self._noise_rng if camera == "overhead" else self._wrist_rng
                image = image + rng.normal(0.0, 1.0, image.shape) * DEPTH_SD_COEF * image**2
        else:
            image = self._renderer.render()
        return image
        

    def _video_frame(self) -> np.ndarray:
        if self.video_size == self.render_size:
            return self.render("front")
        if self._video_renderer is None:
            self._video_renderer = mujoco.Renderer(self.model, *self.video_size)
        self._video_renderer.update_scene(self.data, camera="front")
        return self._video_renderer.render()

    def close(self) -> None:
        if self._video_renderer is not None:
            self._video_renderer.close()
            self._video_renderer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
