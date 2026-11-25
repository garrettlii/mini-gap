"""Results table + per-node failure breakdown from metrics.json files.

Every rate is printed with its Wilson 95% interval (gap.utils.stats.fmt_rate).
Never hand-edit numbers into README or the write-up: regenerate them from here.

  uv run python -m gap.eval.report runs/*/metrics.json            # methods x splits
  uv run python -m gap.eval.report runs/*/metrics.json --by task  # methods x tasks
  uv run python -m gap.eval.report --publish runs/<run> <label>    # -> results/<label>.json
  uv run python -m gap.eval.report --render README.md docs/WRITEUP.md

--render refills marked blocks in place from results/<label>.json (tracked in git):
  <!-- table: <label> [by=task] [methods=a,b] --> ... <!-- /table -->
  <!-- rate: <label> <method> <split> [task] --> ... <!-- /rate -->
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

from gap.utils.runs import read_jsonl
from gap.utils.stats import fmt_rate

RESULTS = Path(__file__).resolve().parents[2] / "results"
BLOCK = re.compile(r"<!-- (table|rate): ([^>]*?) -->.*?<!-- /\1 -->", re.DOTALL)


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
    return _table(load_cells(metrics_paths), by)


def _table(cells: list[dict], by: str = "split", methods: list[str] | None = None) -> str:
    agg: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for c in cells:
        a = agg[(c["method"], c[by])]
        a[0] += c["successes"]
        a[1] += c["n"]
    methods = methods or list(dict.fromkeys(c["method"] for c in cells))
    cols = list(dict.fromkeys(c[by] for c in cells))
    lines = [f"| method | {' | '.join(cols)} |", "|---" * (len(cols) + 1) + "|"]
    for m in methods:
        row = [fmt_rate(*agg[(m, col)]) if agg[(m, col)][1] else "-" for col in cols]
        lines.append(f"| {m} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def _rate(cells: list[dict], method: str, split: str, task: str | None = None) -> str:
    hits = [c for c in cells if c["method"] == method and c["split"] == split
            and task in (None, c["task"])]
    if not hits:
        raise KeyError(f"no cells for {method}/{split}/{task}")
    return fmt_rate(sum(c["successes"] for c in hits), sum(c["n"] for c in hits))


def render(text: str, results_dir: Path = RESULTS) -> str:
    """Refill every marked table/rate block from results/<label>.json."""
    def fill(m: re.Match) -> str:
        kind, spec = m.group(1), m.group(2).split()
        cells = load_cells([str(results_dir / f"{spec[0]}.json")])
        if kind == "rate":
            return f"<!-- rate: {m.group(2)} -->{_rate(cells, *spec[1:])}<!-- /rate -->"
        opts = dict(o.split("=", 1) for o in spec[1:])
        body = _table(cells, opts.get("by", "split"),
                      opts["methods"].split(",") if "methods" in opts else None)
        return f"<!-- table: {m.group(2)} -->\n{body}\n<!-- /table -->"
    return BLOCK.sub(fill, text)


def publish(run_dir: str, label: str, results_dir: Path = RESULTS) -> Path:
    """Copy a run's metrics.json into the tracked results folder under a stable label."""
    results_dir.mkdir(exist_ok=True)
    out = results_dir / f"{label}.json"
    shutil.copy(Path(run_dir) / "metrics.json", out)
    return out


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
    ap.add_argument("metrics", nargs="*")
    ap.add_argument("--by", choices=["split", "task"], default="split")
    ap.add_argument("--publish", nargs=2, metavar=("RUN_DIR", "LABEL"))
    ap.add_argument("--render", nargs="+", metavar="FILE")
    args = ap.parse_args()
    if args.publish:
        print(publish(*args.publish))
    for path in args.render or []:
        Path(path).write_text(render(Path(path).read_text()))
        print("rendered", path)
    if args.metrics:
        print(results_table(args.metrics, args.by))


if __name__ == "__main__":
    main()
