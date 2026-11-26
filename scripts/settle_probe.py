"""How far objects move while reset settles, by how many objects they are stacked on.

  uv run python scripts/settle_probe.py --n 200

Dev seeds, with the settle check switched off, so nothing raises. Per (task, split, level):
the largest displacement and tilt, and how many objects moved more than 1 mm. Evidence for
the reset tolerance (DECISIONS 2026-11-26).
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gap.sim import workcell as wc
from gap.sim.task_sampler import sample
from gap.utils.runs import RunDir
from gap.utils.seeds import seed_range

CASES = [("unstack", "nominal"), ("unstack", "object"), ("invert", "object"),
         ("stack", "object"), ("sort", "object")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    run = RunDir("settle_probe", vars(args))
    wc.SETTLE_TOL = np.inf
    cell = wc.Workcell()
    out = {}
    for task, split in CASES:
        moved, tilt = defaultdict(list), defaultdict(list)
        for seed in range(seed_range("dev").start, seed_range("dev").start + args.n):
            inst = sample(task, seed, split)
            cell.reset(inst)
            for name, pose in inst.object_poses.items():
                level = wc._beneath(name, inst.object_poses)
                moved[level].append(float(np.linalg.norm(cell.data.body(name).xpos - pose.pos)))
                tilt[level].append(float(np.degrees(np.arccos(min(1.0, abs(cell.data.body(name).xmat[8]))))))
        out[f"{task}/{split}"] = {str(k): {"objects": len(v), "max_moved_mm": 1e3 * max(v),
                                           "over_1mm": sum(m > 1e-3 for m in v),
                                           "max_tilt_deg": max(tilt[k])} for k, v in sorted(moved.items())}
        print(task, split, out[f"{task}/{split}"])
    run.metrics(levels=out)


if __name__ == "__main__":
    main()
