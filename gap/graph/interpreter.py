"""Executes a graph on a Workcell and records a per-node trace.

The interpreter never calls an LLM: an optimized graph runs on its own.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from gap.graph.schema import Graph, Node
from gap.sim.success import settled_success
from gap.sim.workcell import Workcell
from gap.skills import registry
from gap.types import EpisodeResult, SkillResult, TaskInstance, TraceEntry

# Phase 6b hook: given (node_id, cell, instance) return "skill", "learned", "fallback" (skill,
# then the learned executor if the skill fails), or "oracle" (try both from the same snapshot,
# keep the skill's result if it succeeded, else the learned one; an in-sim upper bound).
ExecutorPolicy = Callable[[str, Workcell, TaskInstance], str]
# Phase 6b: runs a node with the learned executor instead of skill.run().
LearnedRunner = Callable[[Node, Workcell, dict, dict], SkillResult]


def run_graph(
    graph: Graph,
    cell: Workcell,
    instance: TaskInstance,
    *,
    max_node_visits: int = 60,
    executor_policy: ExecutorPolicy | None = None,
    learned_runner: LearnedRunner | None = None,
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
    (learned_runner) instead of skill.run(). Record which one ran in TraceEntry.executor.
    A learned executor skips the scripted skill's pre-check (that check is the skill's own
    validity gate) but must pass the same post-check.
    """
    cell.frame_sink = frame_sink
    cell.reset(instance)
    outputs: dict[str, dict[str, Any]] = {}
    visits: Counter[str] = Counter()
    retries: Counter[str] = Counter()   # fail-edge traversals per source node
    trace: list[TraceEntry] = []
    node_id: str | None = graph.start_id()
    terminated = "end"
    while node_id is not None:
        if len(trace) >= max_node_visits:
            terminated = "visit_cap"
            break
        node = graph.node(node_id)
        executor = executor_policy(node_id, cell, instance) if executor_policy else "skill"
        entry = _run_node(graph, node, cell, outputs, executor, learned_runner, visits[node_id])
        trace.append(entry)
        visits[node_id] += 1
        if entry.post_ok:
            outputs[node_id] = entry.outputs
            node_id = graph.next_on_success(node_id)
            continue
        fail = graph.fail_edge(node_id)
        if fail is not None and retries[node_id] < fail.max_retries:
            retries[node_id] += 1
            node_id = fail.to
        else:
            terminated = "node_failure"
            break
    success = settled_success(cell)
    cell.frame_sink = None
    return EpisodeResult(instance, success, trace, float(cell.data.time), terminated)


def _run_node(graph, node, cell, outputs, executor, learned_runner, visit) -> TraceEntry:
    try:
        inputs, error = _resolve_inputs(graph, node, outputs), None
    except KeyError as e:
        inputs, error = {}, f"KeyError: {e}"

    def attempt(kind: str) -> TraceEntry:
        learned = None
        if kind == "learned" and learned_runner is not None:
            def learned(c, ins, ps):
                return learned_runner(node, c, ins, ps)
        return run_skill(node.id, node.skill, cell, inputs, node.params, executor=kind,
                         learned=learned, visit=visit, error=error)

    if executor == "fallback":
        entry = attempt("skill")
        return entry if entry.post_ok or error else replace(attempt("learned"), executor="fallback")
    if executor == "oracle":
        snap = cell.snapshot()
        entry = attempt("skill")
        if entry.post_ok or error:
            return entry
        cell.restore(snap)
        return replace(attempt("learned"), executor="oracle")
    return attempt(executor)


def run_skill(node_id: str, skill_name: str, cell: Workcell, inputs: dict[str, Any],
              overrides: dict[str, float], *, executor: str = "skill",
              learned: Callable[[Workcell, dict, dict], SkillResult] | None = None,
              visit: int = 0, error: str | None = None) -> TraceEntry:
    """pre -> run (or the learned executor) -> post for one skill call, as a TraceEntry.

    Shared by graphs and the free-form code baseline so both are judged identically.
    A non-None `error` (e.g. an unresolvable input) records a failure without running.
    """
    before, t0, sim0 = cell.get_state(), time.perf_counter(), cell.data.time
    params: dict[str, float] = {}
    outs: dict[str, Any] = {}
    pre_ok = post_ok = False
    try:
        if error is None:
            skill = registry.get(skill_name)()
            params = skill.resolve_params(overrides)
            pre_ok, why = (True, "") if learned else skill.pre(cell, inputs, params)
            if not pre_ok:
                error = f"pre-check failed: {why}"
            else:
                res = learned(cell, inputs, params) if learned else skill.run(cell, inputs, params)
                outs = res.outputs
                if not res.ok:
                    error = f"run failed: {res.message}"
                else:
                    post_ok, why = skill.post(cell, inputs, outs, params)
                    error = None if post_ok else f"post-check failed: {why}"
    except Exception as e:  # noqa: BLE001 -- any exception is a node failure, not a crash
        error = f"{type(e).__name__}: {e}"
    steps = round((cell.data.time - sim0) / cell.model.opt.timestep)
    return TraceEntry(node_id, skill_name, executor, params, inputs, before, cell.get_state(),
                      outs, pre_ok, post_ok, steps, time.perf_counter() - t0, visit, error)


def _resolve_inputs(graph: Graph, node: Node, outputs: dict) -> dict[str, Any]:
    resolved = {}
    for port, value in node.inputs.items():
        ref = graph.parse_ref(value)
        if ref is None:
            resolved[port] = value
        elif ref[0] not in outputs or ref[1] not in outputs[ref[0]]:
            raise KeyError(f"input {port}: {value} has no value yet ({ref[0]!r} has not succeeded)")
        else:
            resolved[port] = outputs[ref[0]][ref[1]]
    return resolved
