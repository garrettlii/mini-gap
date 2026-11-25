"""Seeded bugs: a reference graph plus one known defect. The loop's repair benchmark.

Each bug is one edit that takes a reference graph (graphs/<task>_example.json) from ~100%
to ~0% on dev seeds; tests check that. Three are parameter bugs for the node specialist, one
is structural for the critic. Nodes are found by skill, so bugs apply to any task's graph.

random_bug() draws training problems from the same parameter families at other magnitudes
and on any matching node (ranges checked to give 0/12 on place). The fixed BUGS stay the
evaluation benchmark.
"""

from __future__ import annotations

import json
import random
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


# family -> (skill, param, value ranges that break the task)
FAMILIES = {
    "diagonal_grasp": ("plan_top_grasp", "yaw_offset", [(0.4, 1.2), (-1.2, -0.4)]),
    "grasp_too_deep": ("plan_top_grasp", "depth", [(-0.015, -0.015)]),
    "tight_verify": ("verify_grasp", "width_max", [(0.02, 0.038)]),
    "high_verify_min": ("verify_grasp", "width_min", [(0.042, 0.06)]),
}


def random_bug(task: str, seed: int) -> tuple[str, SetParam]:
    """(family, edit): a random parameter bug on a random node of the reference graph."""
    rng = random.Random(seed)
    family = rng.choice(sorted(FAMILIES))
    skill, param, ranges = FAMILIES[family]
    node = rng.choice([n.id for n in reference_graph(task).nodes if n.skill == skill])
    lo, hi = rng.choice(ranges)
    return family, SetParam(node=node, param=param, value=round(rng.uniform(lo, hi), 3))


def reference_graph(task: str) -> Graph:
    return Graph.model_validate(json.loads((GRAPHS / f"{task}_example.json").read_text()))


def bug_edit(task: str, bug: str) -> EditOp:
    return BUGS[bug](reference_graph(task))


def buggy_graph(task: str, bug: str) -> Graph:
    return apply_edit(reference_graph(task), bug_edit(task, bug))
