"""Phase 6 data: run the repair loop on many random parameter bugs (place + stack).

  uv run python scripts/gen_repairs.py --per-task 30

Each problem is the reference graph with one random_bug() edit. The loop's specialist is the
4B planner model by default (the teacher); accepted edits become SFT examples and every
specialist query becomes a GRPO prompt. The critic is off: these bugs are all parameter bugs.
One RunDir holds all problems (edits.jsonl, problems.jsonl).
"""

from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gap.agents.llm import local_llm
from gap.agents.node_specialist import propose_fix
from gap.graph.edits import apply_edit
from gap.rehearsal.loop import LoopConfig, improve
from gap.rehearsal.seeded_bugs import random_bug, reference_graph
from gap.utils.runs import RunDir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="place,stack")
    ap.add_argument("--per-task", type=int, default=30)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--specialist", default="planner_repo", choices=["mlx_repo", "planner_repo"])
    ap.add_argument("--n", type=int, default=16)
    args = ap.parse_args()

    loop = LoopConfig("", n_instances=args.n, max_iters=4, patience=2)
    run = RunDir("gen_repairs", {**vars(args), "loop": vars(loop)})
    specialist = functools.partial(propose_fix, local_llm(run, "specialist", model=args.specialist))
    seen = set()
    for task in args.tasks.split(","):
        for seed in range(args.first_seed, args.first_seed + args.per_task):
            family, bug = random_bug(task, seed)
            if (task, bug.node, bug.param, bug.value) in seen:
                continue
            seen.add((task, bug.node, bug.param, bug.value))
            cfg = LoopConfig(task, n_instances=args.n, max_iters=4, patience=2)
            final = improve(apply_edit(reference_graph(task), bug), cfg, specialist,
                            lambda g, t: [], run)
            fixed = final.node(bug.node).params.get(bug.param) != bug.value
            run.append_jsonl("problems.jsonl", {"task": task, "seed": seed, "family": family,
                                                "bug": bug.model_dump(), "changed": fixed})
            print(task, seed, family, bug.node, bug.param, bug.value, "changed" if fixed else "-",
                  flush=True)


if __name__ == "__main__":
    main()
