"""Seed sets. OWNER: done

Rehearsal seeds are what self-learning optimizes on. Eval seeds are only for
Phase 7 and must never be touched by the loop, SFT data, or the arbiter.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "seeds.yaml"


@lru_cache
def _cfg() -> dict:
    return yaml.safe_load(CONFIG.read_text())


def seed_range(name: str) -> range:
    """Named range from configs/seeds.yaml: 'rehearsal', 'dev', or 'eval'."""
    lo, hi = _cfg()["ranges"][name]
    return range(lo, hi)


def eval_seeds(split: str, n: int = 100) -> list[int]:
    """Fixed eval seeds for one split. Same split, same seeds, every run."""
    splits = _cfg()["eval_splits"]
    if split not in splits:
        raise KeyError(f"unknown split {split!r}; known: {splits}")
    base = seed_range("eval").start + splits.index(split) * _cfg()["eval_stride"]
    return list(range(base, base + n))


def rehearsal_seeds(n: int, offset: int = 0) -> list[int]:
    r = seed_range("rehearsal")
    if offset + n > len(r):
        raise ValueError("not enough rehearsal seeds")
    return list(r[offset : offset + n])


def assert_disjoint() -> None:
    names = list(_cfg()["ranges"])
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ra, rb = seed_range(a), seed_range(b)
            if ra.start < rb.stop and rb.start < ra.stop:
                raise AssertionError(f"seed ranges {a} and {b} overlap")
    stride, ev = _cfg()["eval_stride"], seed_range("eval")
    if stride * len(_cfg()["eval_splits"]) > len(ev):
        raise AssertionError("eval splits overflow the eval range")


def is_eval_seed(seed: int) -> bool:
    return seed in seed_range("eval")
