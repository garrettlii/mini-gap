"""SFT data from Phase 5 edit logs.

Output: data/sft/{train,valid}.jsonl in chat format, one example per line:
  {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
Check the exact format your installed mlx-lm-lora expects with `mlx_lm_lora.train --help`.

Rules:
  - Keep only accepted edits that improved success (filtered distillation).
  - Build prompts with gap.agents.node_specialist.build_messages, the same function the live
    agent uses, so train and run-time prompts match exactly.
  - Split by task: train on place + stack; sort is held out entirely.
  - Never include anything derived from eval seeds (gap.utils.seeds.is_eval_seed).
"""

from __future__ import annotations

from pathlib import Path


def build(edit_logs: list[Path], out_dir: Path, train_tasks=("place", "stack"),
          valid_frac: float = 0.1, seed: int = 0) -> dict[str, int]:
    """Return counts per split."""
    raise NotImplementedError
