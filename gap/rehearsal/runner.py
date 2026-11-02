"""Parallel rehearsal.

Each worker process builds its own Workcell (MjModel/MjData are not shareable
across processes). Use multiprocessing with the "spawn" start method on macOS.
"""

from __future__ import annotations

from gap.graph.schema import Graph
from gap.types import EpisodeResult, TaskInstance


def rehearse(graph: Graph, instances: list[TaskInstance], workers: int = 6,
             keep_states: bool = True) -> list[EpisodeResult]:
    """Run graph on every instance; results in the same order as instances.

    Requirements:
      - Refuse eval seeds unless allow_eval=True is passed (add that kwarg): see gap.utils.seeds.
      - Pass graphs as JSON strings to workers; build one Workcell per worker, reuse it.
      - A worker exception becomes an EpisodeResult with terminated_by="exception", never a crash.
      - keep_states=False drops state_before/state_after from traces to save memory in GRPO.
    """
    raise NotImplementedError
