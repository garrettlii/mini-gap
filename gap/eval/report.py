"""Results table + per-node failure breakdown from metrics.json files.

Every rate is printed with its Wilson 95% interval (gap.utils.stats.fmt_rate).
Never hand-edit numbers into README or the write-up: regenerate them from here.

  uv run python -m gap.eval.report runs/*/metrics.json            # methods x splits
  uv run python -m gap.eval.report runs/*/metrics.json --by task  # methods x tasks
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from gap.utils.runs import read_jsonl
from gap.utils.stats import fmt_rate


def load_cells(metrics_paths: list[str]) -> list[dict]:
    """Cells {method, split, task, successes, n} from run_eval or first-shot metrics files."""
    cells: list[dict] = []
    for path in metrics_paths:
        m = json.loads(Path(path).read_text())
        if "cells" in m:
            cells += m["cells"]
        elif "results" in m:  # scripts/first_shot.py: {method: {task: {successes, n}}}
            for method, tasks in m["results"].items():
                for task, c in tasks.items():
                    cells.append({"method": method, "split": m.get("seeds", "dev"), "task": task,
                                  "successes": c["successes"], "n": c["n"]})
    return cells


def results_table(metrics_paths: list[str], by: str = "split") -> str:
    """Markdown table: rows = methods, columns = splits (or tasks), cells = fmt_rate."""
    cells = load_cells(metrics_paths)
    agg: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for c in cells:
        a = agg[(c["method"], c[by])]
        a[0] += c["successes"]
        a[1] += c["n"]
    methods = list(dict.fromkeys(c["method"] for c in cells))
    cols = list(dict.fromkeys(c[by] for c in cells))
    lines = ["| method | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for m in methods:
        row = [fmt_rate(*agg[(m, col)]) if agg[(m, col)][1] else "-" for col in cols]
        lines.append(f"| {m} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def failure_breakdown(episodes_path: str, top: int = 5) -> str:
    """Most common first failing node per method, from an episodes.jsonl."""
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in read_jsonl(Path(episodes_path)):
        if not row["success"]:
            node = (row.get("first_error") or f"({row['terminated_by']})").split(":")[0]
            counts[row["method"]][node] += 1
    out = []
    for method, nodes in counts.items():
        worst = sorted(nodes.items(), key=lambda kv: -kv[1])[:top]
        out.append(f"- {method}: " + ", ".join(f"{n} {k}" for n, k in worst))
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics", nargs="+")
    ap.add_argument("--by", choices=["split", "task"], default="split")
    args = ap.parse_args()
    print(results_table(args.metrics, args.by))


if __name__ == "__main__":
    main()
