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
from gap.types import State, TaskInstance


class Workcell:
    def __init__(self, render_size: tuple[int, int] = (240, 320)):
        self.model: mujoco.MjModel = mb.build_model()
        self.data: mujoco.MjData = mujoco.MjData(self.model)
        self.render_size = render_size          # (height, width)
        self._renderer: mujoco.Renderer | None = None  # create lazily: rehearsal workers rarely render
        self.instance: TaskInstance | None = None

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
        raise NotImplementedError

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
        raise NotImplementedError

    def step(self, ctrl: np.ndarray | None = None, n: int = 1) -> State:
        """Write ctrl (length model.nu: 7 arm joint targets + gripper 0..255) and step n times.

        ctrl=None keeps the previous control. Return get_state() after stepping.
        """
        raise NotImplementedError

    # --------------------------------------------------------- snapshot/restore
    def snapshot(self) -> Any:
        """Exact copy of the physics state (qpos, qvel, act, ctrl, time, warmstart).

        Hint: mujoco.mj_stateSize / mujoco.mj_getState with
        mujoco.mjtState.mjSTATE_INTEGRATION. Test: snapshot, step 200, restore, step 200
        again; both rollouts must match bit for bit.
        """
        raise NotImplementedError

    def restore(self, snap: Any) -> State:
        """Inverse of snapshot(). Remember mj_forward after mj_setState."""
        raise NotImplementedError

    # ---------------------------------------------------------------- render
    def render(self, camera: str = "front", depth: bool = False) -> np.ndarray:
        """RGB uint8 image (H, W, 3), or float depth in meters if depth=True.

        Hints: mujoco.Renderer(model, height, width); update_scene(data, camera=camera);
        enable_depth_rendering() / disable_depth_rendering() to switch modes.
        Cameras: mb.CAMERAS. Note the overhead view is blocked by the arm in the home pose.
        """
        raise NotImplementedError

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
