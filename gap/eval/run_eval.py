"""Run methods x splits on fixed eval seeds.

CLI: uv run python -m gap.eval.run_eval --methods <names> --splits <names> --n 100
Writes one run folder with episodes.jsonl and metrics.json:
  {"cells": [{"method", "split", "task", "successes", "n", "mean_sim_s"}...]}
This is the ONLY code path allowed to use eval seeds.

Methods:
  code, graph_single, graph_multi   policies written by the local planner model (Phase 4),
                                    one per (task, instruction), saved under policies/
  graph_loop                        a loop-repaired graph per task (--loop-graphs <dir>, with
                                    <task>.json files)
  always_skill ... oracle           the Phase 6b arbitration policies on the arbiter graphs
                                    (--models <train_arbiter run>)
The camera noise model is on for every method unless --clean-camera.
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
from gap.rehearsal.runner import map_jobs, rehearse
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
    ap.add_argument("--loop-graphs", default=None, help="dir of <task>.json (graph_loop)")
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
    if method in POLICIES:
        if not args.models:
            raise SystemExit("--models is required for the arbitration policies")
        return rehearse(arbiter_graph(task), insts, keep_states=False, allow_eval=True,
                        camera_noise=noise, policy=(method, args.models))
    if method == "graph_loop":
        graph = Graph.model_validate_json((Path(args.loop_graphs) / f"{task}.json").read_text())
        return rehearse(graph, insts, keep_states=False, allow_eval=True, camera_noise=noise)
    if method not in LLM_METHODS:
        raise SystemExit(f"unknown method {method!r}")
    policies = {i: _llm_policy(method, task, i, args, run) for i in {x.instruction for x in insts}}
    results = [None] * len(insts)
    by_instr: dict[str, list[int]] = {}
    for k, inst in enumerate(insts):
        by_instr.setdefault(inst.instruction, []).append(k)
    for instr, idx in by_instr.items():
        pol = policies[instr]
        sub = [insts[k] for k in idx]
        if pol.get("graph"):
            out = rehearse(Graph.model_validate(pol["graph"]), sub, keep_states=False,
                           allow_eval=True, camera_noise=noise)
        else:
            out = map_jobs(_code_job, [(pol.get("code"), inst, noise) for inst in sub])
        for k, r in zip(idx, out, strict=True):
            results[k] = r
    return results


def _llm_policy(method: str, task: str, instruction: str, args, run) -> dict:
    """Build (or reuse) the method's policy for one instruction; saved as JSON in the run."""
    name = f"policies/{method}/{task}_{hashlib.sha1(instruction.encode()).hexdigest()[:8]}.json"
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
