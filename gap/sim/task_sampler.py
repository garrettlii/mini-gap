"""Randomized task instances.

sample(task, seed, split) must be a pure function: same inputs, same instance,
on any machine. Use np.random.default_rng(seed), never the global RNG.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from gap.sim import model_builder as mb
from gap.types import Pose, TaskInstance

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
    - split != "nominal": apply the overrides from gap/eval/splits.py. Distractors are the
      task's unused cubes, placed in the envelope after the used ones (so the used cubes'
      poses match the same seed without distractors). On the object split every used cube
      gets a variant (features "<cube>_variant", an index into model_builder.VARIANT_NAMES),
      drawn after all poses, and rests at that variant's half height.
    """
    from gap.eval.splits import SPLITS, apply_split  # plain definitions, no sim imports

    rng = np.random.default_rng(seed)
    cfg = task_config() if split == "nominal" else apply_split(task_config(), SPLITS[split])

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

    object_poses = {}
    features = {}
    placed = []
    distractors = [c for c in mb.CUBES if c not in used][: cfg.get("n_distractors", 0)]
    in_envelope = [c for c in mb.CUBES if c in used] + distractors
    for cube in in_envelope + [c for c in mb.CUBES if c not in in_envelope]:
        if cube not in in_envelope:
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

    for name in cfg.get("object_variants", []):
        assert name in mb.VARIANT_NAMES, f"unknown object variant {name!r}"
    for cube in [c for c in mb.CUBES if c in used] if cfg.get("object_variants") else []:
        variant = mb.VARIANT_NAMES.index(str(rng.choice(cfg["object_variants"])))
        p = object_poses[cube]
        object_poses[cube] = Pose((p.pos[0], p.pos[1], mb.half_height(variant)), p.quat)
        features[f"{cube}_variant"] = float(variant)

    stacked = task_cfg.get("start_stacked", [])
    for pair in [stacked] if isinstance(stacked, dict) else stacked:
        # top rests on bottom, in order (towers); the draws above stay unchanged
        top, bottom = pair["top"], pair["bottom"]
        b, t = object_poses[bottom], object_poses[top]
        half = {c: mb.half_height(int(features.get(f"{c}_variant", 0))) for c in (top, bottom)}
        z = b.pos[2] + half[bottom] + half[top]
        object_poses[top] = Pose((b.pos[0], b.pos[1], z), t.quat)
        features.update({f"{top}_x": b.pos[0], f"{top}_y": b.pos[1],
                         f"{top}_distance": features[f"{bottom}_distance"]})

    return TaskInstance(
        task = task,
        instruction = instruction,
        seed = seed,
        object_poses = {c: object_poses[c] for c in mb.CUBES},
        split = split,
        features = features,
    )
