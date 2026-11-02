"""Results table + per-node failure breakdown from metrics.json files.

Every rate is printed with its Wilson 95% interval (gap.utils.stats.fmt_rate).
Never hand-edit numbers into README or the write-up: regenerate them from here.
"""

from __future__ import annotations


def results_table(metrics_paths: list[str]) -> str:
    """Markdown table: rows = methods, columns = splits, cells = fmt_rate."""
    raise NotImplementedError
