"""Phase 6 eval: can each repair model fix held-out random bugs in one edit?

  uv run python scripts/eval_repair.py --adapters sft=runs/<sft>/adapters grpo=runs/<grpo>/adapters

Problems: random_bug() on place and stack (trained tasks) and sort (held out from training),
with bug seeds from 1000 (training used bug seeds below 1000). Each problem: rehearse the
buggy graph on held-out rehearsal seeds, diagnose, and give every model the same prompt for
the buggy node (the node is given, so this measures the edit, not node selection). The model
proposes one edit as in the loop (up to 3 tries with validator feedback); the edited graph is
rehearsed on the same seeds. Problems the buggy graph already solves every time are skipped.

metrics.json cells: successes = problems where the edit raised success, n = problems, plus
mean success before/after, readable by gap/eval/report.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gap.agents.llm import local_llm
from gap.agents.node_specialist import propose_fix
from gap.graph.edits import apply_edit
from gap.rehearsal.diagnose import diagnose
from gap.rehearsal.loop import node_text
from gap.rehearsal.runner import rehearse
from gap.rehearsal.seeded_bugs import random_bug, reference_graph
from gap.sim.task_sampler import sample
from gap.utils.runs import RunDir
from gap.utils.seeds import rehearsal_seeds

HELDOUT_OFFSET = 40_000   # same held-out instance block as scripts/run_loop.py
BUG_SEED0 = 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", nargs="*", default=[], help="name=adapter_dir")
    ap.add_argument("--tasks", default="place,stack,sort")
    ap.add_argument("--problems", type=int, default=30, help="bug seeds per task")
    ap.add_argument("--n", type=int, default=16, help="rehearsal instances per problem")
    args = ap.parse_args()
    run = RunDir("eval_repair", vars(args))
    models = {"zero_shot": None} | dict(a.split("=", 1) for a in args.adapters)
    llms = {m: local_llm(run, f"specialist_{m}", model="mlx_repo", adapter_path=p)
            for m, p in models.items()}
    cells = {}
    for task in args.tasks.split(","):
        insts = [sample(task, s) for s in rehearsal_seeds(args.n, HELDOUT_OFFSET)]
        cache: dict[str, int] = {}

        def score(graph, insts=insts, cache=cache):
            key = graph.to_json()
            if key not in cache:
                cache[key] = sum(r.success for r in rehearse(graph, insts, keep_states=False))
            return cache[key]

        for seed in range(BUG_SEED0, BUG_SEED0 + args.problems):
            family, bug = random_bug(task, seed)
            graph = apply_edit(reference_graph(task), bug)
            results = rehearse(graph, insts)
            before = sum(r.success for r in results)
            if before == args.n:
                continue
            text = node_text(diagnose(results), bug.node)
            for m, llm in llms.items():
                edit = propose_fix(llm, graph, bug.node, text)
                after = score(apply_edit(graph, edit)) if edit else before
                run.append_jsonl("problems.jsonl", {
                    "model": m, "task": task, "seed": seed, "family": family,
                    "bug": bug.model_dump(), "edit": edit.model_dump() if edit else None,
                    "before": before, "after": after, "n": args.n})
                c = cells.setdefault((m, task), {"method": m, "split": "repair", "task": task,
                                                 "successes": 0, "n": 0, "before": [], "after": []})
                c["successes"] += after > before
                c["n"] += 1
                c["before"].append(before / args.n)
                c["after"].append(after / args.n)
                print(task, seed, family, m, before, "->", after, flush=True)
            run.metrics(cells=[_summary(c) for c in cells.values()])


def _summary(c: dict) -> dict:
    return {**{k: c[k] for k in ("method", "split", "task", "successes", "n")},
            "mean_before": float(np.mean(c["before"])), "mean_after": float(np.mean(c["after"]))}


if __name__ == "__main__":
    main()
