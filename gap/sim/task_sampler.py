"""Randomized task instances.

sample(task, seed, split) must be a pure function: same inputs, same instance,
on any machine. Use np.random.default_rng(seed), never the global RNG.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
import numpy as np

from gap.sim import model_builder as mb
from gap.types import TaskInstance, Pose

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
    if split != "nominal":
        raise NotImplementedError("Wait till Phase 7")

    rng = np.random.default_rng(seed)
    cfg = task_config()

    envelope = cfg["envelope"]
    task_cfg = cfg["tasks"][task]
    used = set(task_cfg["objects"])

    instruction = str(rng.choice(task_cfg["instructions"]))

    # Far table corners: on the tabletop (x <= 0.95, |y| <= 0.7), > 1 m from the shoulder
    # (beyond the arm's ~0.96 m reach incl. TCP), clear of both zones and each other.
    parking = {
        "red_cube" : (0.90, 0.60),
        "blue_cube" : (0.90, 0.45),
        "green_cube" : (0.90, -0.60)
    }

    object_poses = dict()
    features = dict()
    placed = list()
    for cube in mb.CUBES:
        if cube not in used:
            x, y = parking[cube]
            object_poses[cube] = Pose((x, y, mb.CUBE_HALF))  # yaw 0: Pose's default quat
        else:
            for _ in range(1000):
                x = float(rng.uniform(*envelope["x"]))
                y = float(rng.uniform(*envelope["y"]))

                if all(np.hypot(x - px, y - py) >= envelope["min_separation"] for px, py in placed):
                    break
            else:
                raise RuntimeError(f"Couldn't place {cube} after 1000 attempts.")

            placed.append((x, y))

            yaw = float(rng.uniform(*envelope["yaw"]))
            folded_yaw = ((yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4)

            features.update({
                f"{cube}_x" : float(x),
                f"{cube}_y" : float(y),
                f"{cube}_yaw" : float(folded_yaw),
                f"{cube}_raw_yaw" : float(yaw),
                f"{cube}_distance" : float(np.hypot(x, y))
            })

            quat = (float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2)))
            object_poses[cube] = Pose((x, y, mb.CUBE_HALF), quat)

    return TaskInstance(
        task = task,
        instruction = instruction,
        seed = seed,
        object_poses = object_poses,
        split = split,
        features = features,
    )
