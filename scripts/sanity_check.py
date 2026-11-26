"""Phase 0 sanity check: physics, rendering, MLX, config. Prints a short report."""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gap.sim import model_builder as mb
from gap.utils.video import save_png

ok = True


def report(name: str, passed: bool, detail: str = "") -> None:
    global ok
    ok &= passed
    print(f"  [{'ok' if passed else 'FAIL'}] {name}{': ' + detail if detail else ''}")


print("Mini-GaP sanity check")
report("python", sys.version_info[:2] >= (3, 11), platform.python_version())
report("mujoco", True, mujoco.__version__)

m = mb.build_model()
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, m.key(mb.HOME_KEY).id)
mujoco.mj_forward(m, d)
z0 = d.body("red_cube").xpos.copy()
for _ in range(500):
    mujoco.mj_step(m, d)
report("physics settles", np.linalg.norm(d.body("red_cube").xpos - z0) < 1e-3,
       f"tcp at {np.round(d.site(mb.TCP_SITE).xpos, 3).tolist()}")

try:
    r = mujoco.Renderer(m, 240, 320)
    frames = []
    for cam in mb.CAMERAS:
        r.update_scene(d, camera=cam)
        frames.append(r.render())
    out = save_png(np.concatenate(frames, axis=1), "runs/sanity/cameras.png")
    report("offscreen render", True, f"wrote {out}")
except Exception as e:  # noqa: BLE001 -- report any failure
    report("offscreen render", False, str(e))

if platform.system() == "Darwin" and platform.machine() == "arm64":
    try:
        import mlx.core as mx
        report("mlx", True, f"default device {mx.default_device()}")
    except Exception as e:  # noqa: BLE001 -- report any failure
        report("mlx", False, str(e))
    gb = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])) // 2**30
    size = "0.8B" if gb < 12 else "2B" if gb < 20 else "4B (4-bit)"
    print(f"  [info] {gb} GB RAM -> train Qwen3.5-{size}")
else:
    print("  [skip] mlx (not Apple Silicon)")

models = yaml.safe_load(Path("configs/models.yaml").read_text())
report("configs/models.yaml local.mlx_repo set", bool(models["local"]["mlx_repo"]), "fill in during Phase 0")
env = Path(".env").read_text() if Path(".env").exists() else ""
report(".env has an API key", any(line.split("=", 1)[-1].strip() for line in env.splitlines()
                                   if "API_KEY" in line))
print("\nAll good." if ok else "\nSome checks failed; see above.")
