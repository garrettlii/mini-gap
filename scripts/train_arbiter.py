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
from gap.arbiter.fallback import FallbackPolicy, collect_demos, mode_of
from gap.arbiter.paired_data import collect
from gap.utils.runs import RunDir, read_jsonl
from gap.utils.seeds import rehearsal_seeds

BC_SEED_OFFSET = 30_000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", type=int, default=600)
    ap.add_argument("--paired", type=int, default=200, help="instances per task")
    ap.add_argument("--tasks", default="place,stack")
    ap.add_argument("--obs", choices=["privileged", "wrist"], default="privileged",
                    help="what the learned grasp sees (DECISIONS 2026-11-26)")
    ap.add_argument("--bc-from", default=None,
                    help="reuse that train_arbiter run's learned grasp; refit only the arbiter")
    args = ap.parse_args()
    bc = _reuse_bc(args) if args.bc_from else None
    run = RunDir("train_arbiter", vars(args))

    if bc is None:
        X, Y, demo_ok = collect_demos(rehearsal_seeds(args.demos, BC_SEED_OFFSET), mode=args.obs)
        bc = FallbackPolicy("grasp", obs_mode=args.obs).fit(X, Y)
        demo_info = {"demo_steps": len(X), "demonstrator_success": demo_ok}
        print(f"demos: {len(X)} steps, demonstrator success {demo_ok:.2f}", flush=True)
    else:
        demo_info = {"bc_from": args.bc_from}
    run.file("bc_grasp.pkl").write_bytes(pickle.dumps(bc))

    for task in args.tasks.split(","):
        n = collect(task, args.paired, run.file("paired.jsonl"), str(run.path))
        print(f"paired rows {task}: {n}", flush=True)
    rows = [r for r in read_jsonl(run.path / "paired.jsonl") if "obs" in r]
    obs = np.array([r["obs"] for r in rows])
    y_s = np.array([r["ok_skill"] for r in rows])
    y_l = np.array([r["ok_learned"] for r in rows])
    arb = Arbiter().fit(obs, y_s, y_l)
    run.file("arbiter.pkl").write_bytes(pickle.dumps(arb))
    # second-try-aware arbiter: success of "skill, then learned" vs "learned, then skill"
    y_ss = np.array([r["ok_skill_then"] for r in rows])
    y_ls = np.array([r["ok_learned_then"] for r in rows])
    run.file("arbiter_seq.pkl").write_bytes(pickle.dumps(Arbiter().fit(obs, y_ss, y_ls)))
    run.metrics(**demo_info, paired_rows=len(rows),
                skill_success=float(y_s.mean()), learned_success=float(y_l.mean()),
                skill_then_learned=float(y_ss.mean()), learned_then_skill=float(y_ls.mean()),
                valid_rate=float(np.mean([r["valid"] for r in rows])),
                coefficients=_coefficients(arb))
    print(json.dumps({"skill": y_s.mean(), "learned": y_l.mean(), "rows": len(rows)}))
    print(run.path)


def _reuse_bc(args) -> FallbackPolicy:
    """Load another run's learned grasp; its config's demos and observation become ours."""
    src = json.loads(Path(args.bc_from, "config.json").read_text())
    bc = pickle.loads(Path(args.bc_from, "bc_grasp.pkl").read_bytes())
    args.demos, args.obs = src["demos"], mode_of(bc)
    return bc


def _coefficients(arb: Arbiter) -> dict:
    out = {}
    for key, m in arb.models.items():
        if not isinstance(m, float):
            coef = m[-1].coef_[0].round(3).tolist()
            out[key] = dict(zip(FEATURES[:len(coef)], coef, strict=True))
    return out


if __name__ == "__main__":
    main()
