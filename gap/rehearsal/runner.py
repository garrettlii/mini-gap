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
             keep_states: bool = True, allow_eval: bool = False,
             camera_noise: bool = False,
             policy: tuple[str, str] | None = None) -> list[EpisodeResult]:
    """Run graph on every instance; results in the same order as instances.

    Requirements:
      - Refuse eval seeds unless allow_eval=True is passed (add that kwarg): see gap.utils.seeds.
      - Pass graphs as JSON strings to workers; build one Workcell per worker, reuse it.
      - A worker exception becomes an EpisodeResult with terminated_by="exception", never a crash.
      - keep_states=False drops state_before/state_after from traces to save memory in GRPO.
      - camera_noise turns on the overhead camera's noise model in every worker's Workcell.
      - policy = (name, models_dir) runs with a Phase 6b arbitration policy
        (gap.arbiter.policies); the instance seed seeds its noise.
    workers <= 1 runs in this process (tests, debugging).
    """
    if not allow_eval and any(is_eval_seed(i.seed) for i in instances):
        raise ValueError("rehearsal on eval seeds; only gap/eval/run_eval.py may pass allow_eval")
    jobs = [(graph.to_json(), inst, keep_states, camera_noise, policy) for inst in instances]
    if workers <= 1:
        return [_run_one(job) for job in jobs]
    return _pool(workers).map(_run_one, jobs)


def map_jobs(fn, jobs: list, workers: int = 6) -> list:
    """fn(job) for every job on the shared spawn pool (fn must be a module-level function)."""
    if workers <= 1:
        return [fn(j) for j in jobs]
    return _pool(workers).map(fn, jobs)


def worker_cell(camera_noise: bool = False):
    """This process's Workcell, built once and reused."""
    global _CELL
    if _CELL is None:
        from gap.sim.workcell import Workcell
        _CELL = Workcell()
    _CELL.camera_noise = camera_noise
    return _CELL


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


def _run_one(job: tuple) -> EpisodeResult:
    graph_json, inst, keep_states, camera_noise, policy = job
    try:
        from gap.graph.interpreter import run_graph
        graph = Graph.model_validate_json(graph_json)
        kw = {}
        if policy is not None:
            from gap.arbiter.policies import executors
            kw["executor_policy"], kw["learned_runner"] = executors(*policy, graph, inst.seed)
        res = run_graph(graph, worker_cell(camera_noise), inst, **kw)
    except Exception:  # noqa: BLE001 -- a broken episode is a result, not a crash
        return EpisodeResult(inst, False, [], 0.0, "exception")
    if not keep_states:
        res.trace = [replace(t, state_before={}, state_after={}) for t in res.trace]
    return res
