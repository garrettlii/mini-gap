"""The policy-authoring methods compared in Phases 4 and 7, behind one interface.

  code          free-form Python over the skills (code-as-policies baseline)
  graph_single  one planner call (with validator feedback) -> graph
  graph_multi   planner -> structure critic review -> graph
"""

from __future__ import annotations

from dataclasses import dataclass, field

from gap.agents.coder import write_code
from gap.agents.critic import review
from gap.agents.llm import local_llm
from gap.agents.planner import plan
from gap.graph.code_exec import run_code
from gap.graph.interpreter import run_graph
from gap.graph.schema import Graph
from gap.sim.workcell import Workcell
from gap.types import EpisodeResult, TaskInstance
from gap.utils.runs import RunDir

METHODS = ("code", "graph_single", "graph_multi")


@dataclass
class Policy:
    method: str
    instruction: str
    graph: Graph | None = None
    code: str | None = None
    log: list[str] = field(default_factory=list)


def build_policy(method: str, instruction: str, run: RunDir | None = None) -> Policy:
    if method == "code":
        code, log = write_code(local_llm(run, role="coder"), instruction)
        return Policy(method, instruction, code=code, log=log)
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; one of {METHODS}")
    graph, log = plan(local_llm(run, role="planner"), instruction)
    if method == "graph_multi" and graph is not None:
        graph, edits = review(local_llm(run, role="critic"), graph)
        log += [f"critic kept: {e.model_dump_json()}" for e in edits]
    return Policy(method, instruction, graph=graph, log=log)


def run_policy(policy: Policy, cell: Workcell, instance: TaskInstance, **kw) -> EpisodeResult:
    if policy.code is not None:
        return run_code(policy.code, cell, instance, **kw)
    if policy.graph is not None:
        return run_graph(policy.graph, cell, instance, **kw)
    cell.reset(instance)
    return EpisodeResult(instance, False, [], 0.0, "no_policy")


def episode_row(policy: Policy, res: EpisodeResult) -> dict:
    """One compact log line per episode (full traces are too big to keep for every run)."""
    fail = next((t for t in res.trace if t.error), None)
    return {"method": policy.method, "task": res.instance.task, "seed": res.instance.seed,
            "instruction": res.instance.instruction, "success": res.success,
            "terminated_by": res.terminated_by, "nodes_run": len(res.trace),
            "first_error": f"{fail.node_id}: {fail.error}" if fail else None}
