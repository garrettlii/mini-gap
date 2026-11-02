"""Phase 3 contracts: validator and interpreter.  uv run pytest --phase 3"""

import json
from pathlib import Path

import pytest

from gap.graph.schema import Graph

pytestmark = pytest.mark.phase(3)

EXAMPLE = json.loads(Path("graphs/place_example.json").read_text())


def codes(graph_dict):
    from gap.graph.validate import validate
    return {e.code for e in validate(Graph.model_validate(graph_dict))}


def mutate(fn):
    g = json.loads(json.dumps(EXAMPLE))
    fn(g)
    return g


def test_example_is_valid():
    assert codes(EXAMPLE) == set()


@pytest.mark.parametrize("fn,code", [
    (lambda g: g["nodes"][0].update(skill="teleport"), "unknown_skill"),
    (lambda g: g["nodes"][1]["params"].update(approach_height=9.0), "param_range"),
    (lambda g: g["nodes"][1]["params"].update(warp=1.0), "unknown_param"),
    (lambda g: g["nodes"][1]["in"].pop("pose"), "missing_input"),
    (lambda g: g["nodes"][1]["in"].update(pose="find.color"), "bad_ref"),
    (lambda g: g["nodes"][2]["in"].update(pose="plan.missing"), "bad_ref"),
    (lambda g: g["nodes"][1]["in"].update(pose="above.pose"), "forward_ref"),
    (lambda g: g["edges"].append({"from": "find", "to": "ghost"}), "bad_edge"),
    (lambda g: g["nodes"].append(dict(g["nodes"][0])), "duplicate_id"),
    (lambda g: g.update(start="ghost"), "bad_start"),
])
def test_validator_catches(fn, code):
    assert code in codes(mutate(fn))


def test_interpreter_traces_every_visit():
    from gap.graph.interpreter import run_graph
    from gap.sim.task_sampler import sample
    from gap.sim.workcell import Workcell
    cell = Workcell()
    res = run_graph(Graph.model_validate(EXAMPLE), cell, sample("place", seed=50010))
    assert res.trace and res.trace[0].node_id == "find"
    assert all(t.state_before and t.state_after for t in res.trace)
