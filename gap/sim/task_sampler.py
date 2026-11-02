"""Randomized task instances.

sample(task, seed, split) must be a pure function: same inputs, same instance,
on any machine. Use np.random.default_rng(seed), never the global RNG.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from gap.types import TaskInstance

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "tasks.yaml"


@lru_cache
def task_config() -> dict:
    return yaml.safe_load(CONFIG.read_text())


def sample(task: str, seed: int, split: str = "nominal") -> TaskInstance:
    """Sample object poses inside the envelope with rejection on min_separation.

    - Objects not used by the task go to a parking spot off the table edge (or stay out of
      reach) so they never interfere.
    - Cube z = CUBE_HALF (resting on the table). Yaw -> quaternion about world z.
    - Fill instance.features with the variables diagnosis and the arbiter will need:
      per-object x, y, yaw, and distance from the robot base.
    - split != "nominal": apply the overrides from gap/eval/splits.py (Phase 7). Until then,
      raise NotImplementedError for other splits rather than silently ignoring them.
    """
    raise NotImplementedError
