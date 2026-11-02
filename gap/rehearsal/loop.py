"""Self-learning loop: rehearse -> diagnose -> edit -> rehearse.

Log every step to edits.jsonl as {graph, report, edit, success_before, success_after,
accepted}. These records are the Phase 6 training data, so log rejected edits too.
"""

from __future__ import annotations

from dataclasses import dataclass

from gap.graph.schema import Graph
from gap.utils.runs import RunDir


@dataclass
class LoopConfig:
    task: str
    n_instances: int = 32
    max_iters: int = 12
    patience: int = 3            # stop after this many iterations without improvement
    workers: int = 6


def improve(graph: Graph, cfg: LoopConfig, specialist, critic, run: RunDir) -> Graph:
    """Hill-climb on rehearsal seeds only. Keep an edit only if success improves.

    Route by failure category: parameter/skill problems -> specialist for that node;
    ordering/missing-step problems -> critic. Validate every edited graph before rehearsing.
    """
    raise NotImplementedError
