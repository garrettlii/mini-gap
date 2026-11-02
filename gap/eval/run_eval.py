"""Run methods x splits on fixed eval seeds.

CLI: uv run python -m gap.eval.run_eval --methods <names> --splits <names> --n 100
Writes one run folder with episodes.jsonl and metrics.json:
  {"cells": [{"method", "split", "task", "successes", "n", "mean_sim_s"}...]}
This is the ONLY code path allowed to use eval seeds.
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
