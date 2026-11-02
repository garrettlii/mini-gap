"""Executes a graph on a Workcell and records a per-node trace.

The interpreter never calls an LLM: an optimized graph runs on its own.
"""

from __future__ import annotations

from typing import Callable

from gap.graph.schema import Graph
from gap.sim.workcell import Workcell
from gap.types import EpisodeResult, TaskInstance

# Phase 6b hook: given (node_id, cell, instance) return "skill" or "learned".
ExecutorPolicy = Callable[[str, Workcell, TaskInstance], str]


def run_graph(
    graph: Graph,
    cell: Workcell,
    instance: TaskInstance,
    *,
    max_node_visits: int = 60,
    executor_policy: ExecutorPolicy | None = None,
    frame_sink: list | None = None,
) -> EpisodeResult:
    """Reset the cell to the instance, walk the graph, return an EpisodeResult.

    Loop:
      1. resolve inputs: constants as-is, references from the outputs dict of earlier nodes
      2. resolve params: skill.resolve_params(node.params)
      3. pre-check; if ok, run (wrapped in try/except: an exception is a node failure)
      4. post-check; append a TraceEntry with state_before / state_after
      5. choose the next node per the semantics in gap/graph/schema.py, counting retries per
         fail edge; stop at the end, on unrecovered failure, or at max_node_visits
      6. success = gap.sim.success.check(...) on the final state  (never trust the graph)

    executor_policy is None until Phase 6b; then "learned" runs the node's fallback policy
    instead of skill.run(). Record which one ran in TraceEntry.executor.
    """
    raise NotImplementedError
