"""Eval splits: each changes ONE thing from nominal. Definitions OWNER: done"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Split:
    name: str
    tests: str
    envelope_scale: float = 1.0          # multiply envelope x/y half-widths
    full_yaw: bool = False
    object_variants: tuple[str, ...] = ()  # e.g. ("cube_3cm", "cube_5cm", "cylinder")
    paraphrase_instructions: bool = False
    n_distractors: int = 0
    tasks: tuple[str, ...] = ()          # empty = all trained tasks
    extra: dict = field(default_factory=dict)


SPLITS: dict[str, Split] = {
    "nominal": Split("nominal", "overfitting to rehearsal seeds"),
    "pose": Split("pose", "robustness to placement", envelope_scale=2.0, full_yaw=True),
    "object": Split("object", "grasp generalization",
                    object_variants=("cube_3cm", "cube_5cm", "cylinder")),
    "instruction": Split("instruction", "language grounding", paraphrase_instructions=True),
    "distractors": Split("distractors", "perception", n_distractors=2),
    "heldout_task": Split("heldout_task", "transfer", tasks=("sort",)),
}


def apply_split(task_cfg: dict, split: Split) -> dict:
    """Return a modified copy of the task config for this split (used by task_sampler).

    task_cfg is the whole configs/tasks.yaml dict. The sampler reads two extra keys this adds:
    n_distractors and object_variants (variant geoms built by model_builder._add_variants).
    """
    cfg = copy.deepcopy(task_cfg)
    env = cfg["envelope"]
    for axis in ("x", "y"):
        lo, hi = env[axis]
        mid, half = (lo + hi) / 2, (hi - lo) / 2 * split.envelope_scale
        env[axis] = [mid - half, mid + half]
    if split.full_yaw:
        env["yaw"] = [-math.pi, math.pi]
    if split.paraphrase_instructions:
        for task in cfg["tasks"].values():
            task["instructions"] = task["paraphrases"]
    cfg["n_distractors"] = split.n_distractors
    cfg["object_variants"] = list(split.object_variants)
    return cfg
