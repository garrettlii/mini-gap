"""Run methods x splits on fixed eval seeds.

CLI: uv run python -m gap.eval.run_eval --methods <names> --splits <names> --n 100
Writes one run folder with episodes.jsonl and metrics.json:
  {"cells": [{"method", "split", "task", "successes", "n", "mean_sim_s"}...]}
This is the ONLY code path allowed to use eval seeds.

Methods:
  code, graph_single, graph_multi   policies written by the local planner model (Phase 4),
                                    one per (task, instruction), saved under policies/
  graph_loop                        the loop-repaired graph per task (--loop-runs: run_loop
                                    run dirs; each one's final_graph.json, keyed by its task)
  always_skill ... oracle           the Phase 6b arbitration policies on the arbiter graphs
                                    (--models <train_arbiter run>)
  <graph method>+<policy>           a graph method run under an arbitration policy, e.g.
                                    graph_loop+learned_arbiter
The camera noise model is on for every method unless --clean-camera. Every policy runs with
the RGB-D locator: the ground-truth locator (privileged) is swapped out at evaluation.
--dev swaps in dev seeds (a block of 1000 per split) for tuning before the real eval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from gap.arbiter.policies import POLICIES, arbiter_graph
from gap.eval.splits import SPLITS
from gap.graph.schema import Graph
from gap.rehearsal.runner import map_jobs, rehearse, with_rgbd
from gap.sim.task_sampler import sample
from gap.utils.runs import RunDir
from gap.utils.seeds import eval_seeds, seed_range

LLM_METHODS = ("code", "graph_single", "graph_multi")
TRAINED_TASKS = ("place", "stack")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--models", default=None, help="train_arbiter run dir (arbiter methods)")
    ap.add_argument("--loop-runs", nargs="*", default=[], help="run_loop dirs (graph_loop)")
    ap.add_argument("--policies-from", default=None, help="reuse LLM policies from a run dir")
    ap.add_argument("--clean-camera", action="store_true")
    ap.add_argument("--dev", action="store_true", help="dev seeds instead of eval seeds")
    args = ap.parse_args(argv)
    run = RunDir("eval", vars(args))
    noise = not args.clean_camera
    cells = []
    for split in args.splits:
        for task in SPLITS[split].tasks or TRAINED_TASKS:
            insts = [sample(task, s, split) for s in _seeds(split, args.n, args.dev)]
            for method in args.methods:
                results = _run_method(method, task, insts, noise, args, run)
                for r in results:
                    run.append_jsonl("episodes.jsonl", _row(method, split, r))
                cells.append({"method": method, "split": split, "task": task,
                              "successes": sum(r.success for r in results), "n": len(results),
                              "mean_sim_s": float(np.mean([r.sim_time_s for r in results]))})
                print(f"{split:12s} {task:6s} {method:20s} {cells[-1]['successes']}/{len(results)}",
                      flush=True)
                run.metrics(cells=cells, camera_noise=noise)


def _seeds(split: str, n: int, dev: bool) -> list[int]:
    if not dev:
        return eval_seeds(split, n)
    start = seed_range("dev").start + list(SPLITS).index(split) * 1000
    return list(range(start, start + n))


def _run_method(method, task, insts, noise, args, run):
    """method is a name from the module docstring, optionally "<graph method>+<policy>"."""
    base, _, arb = method.partition("+")
    if (base in POLICIES or arb) and not args.models:
        raise SystemExit("--models is required for the arbitration policies")
    if base in POLICIES:
        return _rehearse(arbiter_graph(task), insts, noise, (base, args.models))
    policy = (arb, args.models) if arb else None
    if base == "graph_loop":
        return _rehearse(_loop_graph(args.loop_runs, task), insts, noise, policy)
    if base not in LLM_METHODS:
        raise SystemExit(f"unknown method {method!r}")
    results = [None] * len(insts)
    by_instr: dict[str, list[int]] = {}
    for k, inst in enumerate(insts):
        by_instr.setdefault(inst.instruction, []).append(k)
    for instr, idx in by_instr.items():
        pol = _llm_policy(base, task, instr, args, run)
        sub = [insts[k] for k in idx]
        if pol.get("graph"):
            out = _rehearse(Graph.model_validate(pol["graph"]), sub, noise, policy)
        elif policy:
            raise SystemExit(f"{method}: arbitration needs a graph, not code")
        else:
            code = (pol.get("code") or "").replace("locate_object_gt", "locate_object_rgbd") or None
            out = map_jobs(_code_job, [(code, inst, noise) for inst in sub])
        for k, r in zip(idx, out, strict=True):
            results[k] = r
    return results


def _rehearse(graph: Graph, insts, noise: bool, policy=None):
    """Eval rehearsal; the privileged ground-truth locator is not available at deployment."""
    return rehearse(with_rgbd(graph), insts, keep_states=False, allow_eval=True, camera_noise=noise,
                    policy=policy)


def _loop_graph(run_dirs: list[str], task: str) -> Graph:
    for d in run_dirs:
        if json.loads(Path(d, "config.json").read_text())["task"] == task:
            return Graph.model_validate_json(Path(d, "final_graph.json").read_text())
    raise SystemExit(f"no --loop-runs dir for task {task!r}")


def _llm_policy(method: str, task: str, instruction: str, args, run) -> dict:
    """Build (or reuse) the method's policy for one instruction; saved as JSON in the run."""
    name = f"policies/{method}/{task}_{hashlib.sha1(instruction.encode()).hexdigest()[:8]}.json"
    if (run.path / name).exists():   # built for an earlier split in this run
        return json.loads((run.path / name).read_text())
    if args.policies_from:
        for p in Path(args.policies_from, "policies", method).glob(f"{task}_*.json"):
            pol = json.loads(p.read_text())
            if pol["instruction"] == instruction:
                run.write_json(name, pol)
                return pol
    from gap.agents.methods import build_policy
    pol = build_policy(method, instruction, run)
    out = {"instruction": instruction, "log": pol.log, "code": pol.code,
           "graph": json.loads(pol.graph.to_json()) if pol.graph else None}
    run.write_json(name, out)
    return out


def _code_job(job):
    from gap.graph.code_exec import run_code
    from gap.rehearsal.runner import worker_cell
    from gap.types import EpisodeResult
    code, inst, noise = job
    cell = worker_cell(noise)
    if not code:
        cell.reset(inst)
        return EpisodeResult(inst, False, [], 0.0, "no_policy")
    try:
        res = run_code(code, cell, inst)
    except Exception:  # noqa: BLE001 -- a crashing policy is a failed episode
        return EpisodeResult(inst, False, [], 0.0, "exception")
    res.trace = []
    return res


def _row(method: str, split: str, r) -> dict:
    fail = next((t for t in r.trace if t.error), None)
    return {"method": method, "split": split, "task": r.instance.task, "seed": r.instance.seed,
            "success": r.success, "sim_s": r.sim_time_s, "terminated_by": r.terminated_by,
            "executors": sorted({t.executor for t in r.trace}),
            "first_error": f"{fail.node_id}: {fail.error}" if fail else None,
            "features": r.instance.features}


if __name__ == "__main__":
    main()
