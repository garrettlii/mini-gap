"""Video helpers. OWNER: done"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def save_mp4(frames: list[np.ndarray], path: str | Path, fps: int = 30) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=7) as w:
        for f in frames:
            w.append_data(f)
    return path


def save_png(frame: np.ndarray, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, frame)
    return path
