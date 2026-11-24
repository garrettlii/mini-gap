"""Parallel rehearsal.

Each worker process builds its own Workcell (MjModel/MjData are not shareable
across processes). Use multiprocessing with the "spawn" start method on macOS.
"""

from __future__ import annotations

import atexit
import multiprocessing as mp
from dataclasses import replace

from gap.graph.schema import Graph
from gap.types import EpisodeResult, TaskInstance
from gap.utils.seeds import is_eval_seed

_POOL: tuple[int, mp.pool.Pool] | None = None   # (workers, pool), reused across calls
_CELL = None                                     # one Workcell per worker process


def rehearse(graph: Graph, instances: list[TaskInstance], workers: int = 6,
             keep_states: bool = True, allow_eval: bool = False) -> list[EpisodeResult]:
    """Run graph on every instance; results in the same order as instances.

    Requirements:
      - Refuse eval seeds unless allow_eval=True is passed (add that kwarg): see gap.utils.seeds.
      - Pass graphs as JSON strings to workers; build one Workcell per worker, reuse it.
      - A worker exception becomes an EpisodeResult with terminated_by="exception", never a crash.
      - keep_states=False drops state_before/state_after from traces to save memory in GRPO.
    workers <= 1 runs in this process (tests, debugging).
    """
    if not allow_eval and any(is_eval_seed(i.seed) for i in instances):
        raise ValueError("rehearsal on eval seeds; only gap/eval/run_eval.py may pass allow_eval")
    jobs = [(graph.to_json(), inst, keep_states) for inst in instances]
    if workers <= 1:
        return [_run_one(job) for job in jobs]
    return _pool(workers).map(_run_one, jobs)


def _pool(workers: int) -> mp.pool.Pool:
    global _POOL
    if _POOL is None or _POOL[0] != workers:
        if _POOL is not None:
            _POOL[1].terminate()
        _POOL = (workers, mp.get_context("spawn").Pool(workers))
    return _POOL[1]


@atexit.register
def _close_pool() -> None:
    if _POOL is not None:
        _POOL[1].terminate()


def _run_one(job: tuple[str, TaskInstance, bool]) -> EpisodeResult:
    global _CELL
    graph_json, inst, keep_states = job
    try:
        from gap.graph.interpreter import run_graph
        from gap.sim.workcell import Workcell
        if _CELL is None:
            _CELL = Workcell()
        res = run_graph(Graph.model_validate_json(graph_json), _CELL, inst)
    except Exception:  # noqa: BLE001 -- a broken episode is a result, not a crash
        return EpisodeResult(inst, False, [], 0.0, "exception")
    if not keep_states:
        res.trace = [replace(t, state_before={}, state_after={}) for t in res.trace]
    return res
