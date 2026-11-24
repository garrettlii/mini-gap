"""Run the Place oracle on dev seeds, log success, and save an MP4 per seed.

  uv run python scripts/record_oracle.py 50000 50001
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gap.sim.oracle import run_place_oracle
from gap.sim.task_sampler import sample
from gap.sim.workcell import VIDEO_FPS, Workcell
from gap.utils.runs import RunDir
from gap.utils.video import save_mp4

seeds = [int(a) for a in sys.argv[1:]] or [50000]
run = RunDir("oracle_video", {"task": "place", "seeds": seeds})
cell = Workcell(render_size=(480, 640))
wins = 0
for seed in seeds:
    frames: list = []
    ok = run_place_oracle(cell, sample("place", seed=seed), frame_sink=frames)
    wins += ok
    run.append_jsonl("episodes.jsonl", {"seed": seed, "success": ok})
    print(seed, "success" if ok else "FAIL", save_mp4(frames, run.file(f"seed{seed}.mp4"), VIDEO_FPS))
run.metrics(task="place", method="oracle", successes=wins, n=len(seeds))
