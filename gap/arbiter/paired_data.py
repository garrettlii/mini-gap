"""Paired executor outcomes from identical states.

For each rehearsal instance and each arbitrated node:
  snap = cell.snapshot() before the node
  run the scripted skill  -> post_ok_skill;   cell.restore(snap)
  run the learned policy  -> post_ok_learned; cell.restore(snap)
  record features at snap + both labels
Pairs from the same state remove most confounding between "which executor" and
"which situation", which is what makes the arbiter result credible.
"""

from __future__ import annotations

from pathlib import Path


def collect(task: str, n_instances: int, out_path: Path) -> int:
    """Write rows to a parquet/CSV/JSONL file; return the row count."""
    raise NotImplementedError
