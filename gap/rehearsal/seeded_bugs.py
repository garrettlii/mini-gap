"""Seeded bugs: a reference graph plus one known defect. The loop's repair benchmark.

Each bug is one edit that takes a reference graph (graphs/<task>_example.json) from ~100%
to ~0% on dev seeds; tests check that. Three are parameter bugs for the node specialist, one
is structural for the critic. Nodes are found by skill, so bugs apply to any task's graph.
"""

from __future__ import annotations

import json
from pathlib import Path

from gap.graph.edits import EditOp, RemoveNode, SetParam, apply_edit
from gap.graph.schema import Graph

GRAPHS = Path(__file__).resolve().parents[2] / "graphs"


def _first(graph: Graph, skill: str) -> str:
    return next(n.id for n in graph.nodes if n.skill == skill)


def _last(graph: Graph, skill: str) -> str:
    return [n.id for n in graph.nodes if n.skill == skill][-1]


BUGS = {
    # fingers close across the cube's diagonal: grasp width ~5 cm, verify_grasp rejects it
    "diagonal_grasp": lambda g: SetParam(node=_first(g, "plan_top_grasp"), param="yaw_offset",
                                         value=0.785),
    # grasp pose 1.5 cm below the cube center: fingertips jam on the table
    "grasp_too_deep": lambda g: SetParam(node=_first(g, "plan_top_grasp"), param="depth",
                                         value=-0.015),
    # verification band excludes a real 4 cm grasp
    "tight_verify": lambda g: SetParam(node=_first(g, "verify_grasp"), param="width_max",
                                       value=0.02),
    # the final release is missing: the cube leaves with the hand
    "no_release": lambda g: RemoveNode(node=_last(g, "open_gripper")),
}


def reference_graph(task: str) -> Graph:
    return Graph.model_validate(json.loads((GRAPHS / f"{task}_example.json").read_text()))


def bug_edit(task: str, bug: str) -> EditOp:
    return BUGS[bug](reference_graph(task))


def buggy_graph(task: str, bug: str) -> Graph:
    return apply_edit(reference_graph(task), bug_edit(task, bug))
