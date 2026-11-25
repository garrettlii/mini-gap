"""Phase 5: run the self-learning loop on one task from a seeded-bug or planner graph.

  uv run python scripts/run_loop.py --task place --bug diagonal_grasp
  uv run python scripts/run_loop.py --task place --start planner

Optimizes on one block of rehearsal seeds, then measures before/after on a disjoint held-out
block of rehearsal seeds. Writes edits.jsonl (Phase 6 data) and metrics.json.
"""

from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gap.agents.critic import propose_structure
from gap.agents.llm import local_llm
from gap.agents.node_specialist import propose_fix
from gap.agents.planner import plan
from gap.rehearsal.loop import LoopConfig, improve
from gap.rehearsal.runner import rehearse
from gap.rehearsal.seeded_bugs import BUGS, buggy_graph
from gap.sim.task_sampler import sample, task_config
from gap.utils.runs import RunDir
from gap.utils.seeds import rehearsal_seeds

HELDOUT_OFFSET = 40_000   # held-out block of rehearsal seeds, disjoint from the loop's block


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--bug", choices=sorted(BUGS))
    ap.add_argument("--start", choices=["bug", "planner"], default="bug")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--heldout", type=int, default=64)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--adapter", default=None, help="LoRA adapter for the specialist (Phase 6)")
    ap.add_argument("--specialist", default="mlx_repo", choices=["mlx_repo", "planner_repo"],
                    help="configs/models.yaml key: the 2B repair model or the 4B planner model")
    args = ap.parse_args()

    cfg = LoopConfig(args.task, n_instances=args.n, max_iters=args.iters,
                     seed_offset=args.seed_offset)
    name = f"loop_{args.task}_{args.bug or args.start}_{args.specialist.split('_')[0]}"
    run = RunDir(name, {**vars(args), "loop": vars(cfg)})
    if args.start == "planner":
        instruction = task_config()["tasks"][args.task]["instructions"][0]
        graph, log = plan(local_llm(run, "planner", model="planner_repo"), instruction)
        if graph is None or log[-1] != "valid":
            run.metrics(task=args.task, start="planner", error="planner produced no valid graph")
            print("planner produced no valid graph:", log)
            return
    else:
        graph = buggy_graph(args.task, args.bug)
    specialist = functools.partial(propose_fix, local_llm(run, "specialist", model=args.specialist,
                                                          adapter_path=args.adapter))
    critic = functools.partial(propose_structure, local_llm(run, "critic", model="planner_repo"))
    final = improve(graph, cfg, specialist, critic, run)

    held = [sample(args.task, s) for s in rehearsal_seeds(args.heldout, HELDOUT_OFFSET)]
    before = sum(r.success for r in rehearse(graph, held, keep_states=False))
    after = sum(r.success for r in rehearse(final, held, keep_states=False))
    run.write_json("final_graph.json", final.model_dump(by_alias=True, exclude_none=True))
    run.metrics(task=args.task, start=args.start, bug=args.bug, heldout_n=args.heldout,
                heldout_before=before, heldout_after=after)
    print(f"{args.task} {args.bug or args.start}: held-out {before}/{args.heldout} -> "
          f"{after}/{args.heldout}  ({run.path})")


if __name__ == "__main__":
    main()
