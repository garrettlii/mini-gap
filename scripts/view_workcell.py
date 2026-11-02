"""Interactive viewer. On macOS run with mjpython:  uv run mjpython scripts/view_workcell.py

Holds the arm at the home keyframe so you can inspect the scene, toggle site/geom
groups (group 3 shows the TCP site), and drag objects with Ctrl+right-click.
"""

import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gap.sim import model_builder as mb  # noqa: E402

m = mb.build_model()
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, m.key(mb.HOME_KEY).id)

with mujoco.viewer.launch_passive(m, d) as v:
    while v.is_running():
        t0 = time.time()
        mujoco.mj_step(m, d)
        v.sync()
        time.sleep(max(0.0, m.opt.timestep - (time.time() - t0)))
