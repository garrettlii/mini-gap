"""Eval splits: each changes ONE thing from nominal. Definitions OWNER: done"""

from __future__ import annotations

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

    'object' needs extra bodies in the MJCF (or MjSpec edits at build time): plan that in
    Phase 7, not before.
    """
    raise NotImplementedError
