"""Phase 4: first-shot success of each authoring method on each task (dev seeds).

  uv run python scripts/first_shot.py --seeds 20

Writes runs/<stamp>_first_shot/: policies/, episodes.jsonl, llm_calls.jsonl, metrics.json.
Dev seeds only; the paper's table comes from gap/eval/run_eval.py on eval seeds.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gap.agents.methods import METHODS, build_policy, episode_row, run_policy
from gap.sim.task_sampler import sample, task_config
from gap.sim.workcell import Workcell
from gap.utils.runs import RunDir
from gap.utils.seeds import seed_range

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=20)
ap.add_argument("--tasks", nargs="+", default=["lift", "place", "stack", "sort"])
ap.add_argument("--methods", nargs="+", default=list(METHODS))
args = ap.parse_args()

seeds = seed_range("dev")[: args.seeds]
run = RunDir("first_shot", {"seeds": "dev", "n_seeds": len(seeds), "tasks": args.tasks,
                            "methods": args.methods})
cell = Workcell()
results: dict[str, dict[str, dict[str, int]]] = {m: {} for m in args.methods}
for task in args.tasks:
    instructions = task_config()["tasks"][task]["instructions"]
    for method in args.methods:
        policies = {}
        for i, instr in enumerate(instructions):
            pol = build_policy(method, instr, run)
            policies[instr] = pol
            run.write_json(f"policies/{method}/{task}_{i}.json", {
                "instruction": instr, "log": pol.log, "code": pol.code,
                "graph": pol.graph.model_dump(by_alias=True, exclude_none=True) if pol.graph else None})
        wins = 0
        for seed in seeds:
            inst = sample(task, seed)
            res = run_policy(policies[inst.instruction], cell, inst)
            wins += res.success
            run.append_jsonl("episodes.jsonl", episode_row(policies[inst.instruction], res))
        results[method][task] = {"successes": wins, "n": len(seeds)}
        print(f"{method:13s} {task:6s} {wins}/{len(seeds)}", flush=True)
run.metrics(phase=4, seeds="dev", results=results)
print("wrote", run.path)
