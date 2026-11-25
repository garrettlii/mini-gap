"""Phase 6b: train the learned grasp executor and the arbiter, all on rehearsal seeds.

  uv run python scripts/train_arbiter.py --demos 600 --paired 200

1. Demonstrator grasps from perturbed hand poses -> behavior-cloned MLP (bc_grasp.pkl).
2. Paired outcomes: arbiter graphs on place + stack (camera noise on, nominal and object
   variants); at every close_gripper node both executors run from the same snapshot.
3. Arbiter (one logistic regression per executor) on the paired rows (arbiter.pkl).
Writes everything to one RunDir; pass its path to run_eval.py --models.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gap.arbiter.arbiter import FEATURES, Arbiter
from gap.arbiter.fallback import FallbackPolicy, collect_demos
from gap.arbiter.paired_data import collect
from gap.utils.runs import RunDir, read_jsonl
from gap.utils.seeds import rehearsal_seeds

BC_SEED_OFFSET = 30_000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", type=int, default=600)
    ap.add_argument("--paired", type=int, default=200, help="instances per task")
    ap.add_argument("--tasks", default="place,stack")
    args = ap.parse_args()
    run = RunDir("train_arbiter", vars(args))

    X, Y, demo_ok = collect_demos(rehearsal_seeds(args.demos, BC_SEED_OFFSET))
    bc = FallbackPolicy("grasp").fit(X, Y)
    run.file("bc_grasp.pkl").write_bytes(pickle.dumps(bc))
    print(f"demos: {len(X)} steps, demonstrator success {demo_ok:.2f}", flush=True)

    for task in args.tasks.split(","):
        n = collect(task, args.paired, run.file("paired.jsonl"), str(run.path))
        print(f"paired rows {task}: {n}", flush=True)
    rows = [r for r in read_jsonl(run.path / "paired.jsonl") if "obs" in r]
    obs = np.array([r["obs"] for r in rows])
    y_s = np.array([r["ok_skill"] for r in rows])
    y_l = np.array([r["ok_learned"] for r in rows])
    arb = Arbiter().fit(obs, y_s, y_l)
    run.file("arbiter.pkl").write_bytes(pickle.dumps(arb))
    run.metrics(demo_steps=len(X), demonstrator_success=demo_ok, paired_rows=len(rows),
                skill_success=float(y_s.mean()), learned_success=float(y_l.mean()),
                valid_rate=float(np.mean([r["valid"] for r in rows])),
                coefficients=_coefficients(arb))
    print(json.dumps({"skill": y_s.mean(), "learned": y_l.mean(), "rows": len(rows)}))
    print(run.path)


def _coefficients(arb: Arbiter) -> dict:
    out = {}
    for key, m in arb.models.items():
        if not isinstance(m, float):
            out[key] = dict(zip(FEATURES, m[-1].coef_[0].round(3).tolist(), strict=True))
    return out


if __name__ == "__main__":
    main()
