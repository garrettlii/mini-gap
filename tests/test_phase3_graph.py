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


@pytest.mark.parametrize("task", ["stack", "sort"])
def test_other_example_graphs_are_valid(task):
    assert codes(json.loads(Path(f"graphs/{task}_example.json").read_text())) == set()


def test_apply_edit_ops_return_new_graphs():
    from gap.graph.edits import apply_edit, parse_edit
    g = Graph.model_validate(EXAMPLE)
    before = g.to_json()
    g2 = apply_edit(g, parse_edit({"op": "set_param", "node": "close", "param": "force", "value": 0.9}))
    assert g2.node("close").params["force"] == 0.9 and g.to_json() == before, "input untouched"
    g3 = apply_edit(g, parse_edit({"op": "insert_after", "after": "close",
                                   "new": {"id": "wait", "skill": "open_gripper"}}))
    assert g3.ids()[g3.ids().index("close") + 1] == "wait"
    g4 = apply_edit(g, parse_edit({"op": "swap_skill", "node": "find", "skill": "locate_object_rgbd"}))
    assert g4.node("find").skill == "locate_object_rgbd"
    g5 = apply_edit(g, parse_edit({"op": "add_edge", "edge": {"from": "reach", "to": "above",
                                                               "on": "fail", "max_retries": 1}}))
    assert g5.fail_edge("reach").to == "above"


def test_remove_node_drops_edges_and_leaves_refs_for_the_validator():
    from gap.graph.edits import apply_edit, parse_edit
    from gap.graph.validate import validate
    g = apply_edit(Graph.model_validate(EXAMPLE), parse_edit({"op": "remove_node", "node": "find"}))
    assert "find" not in g.ids() and not any("find" in (e.src, e.to) for e in g.edges)
    assert any(e.code == "bad_ref" and e.node == "plan" for e in validate(g))


def test_apply_edit_rejects_unknown_nodes():
    from gap.graph.edits import apply_edit, parse_edit
    with pytest.raises(ValueError, match="no node 'ghost'"):
        apply_edit(Graph.model_validate(EXAMPLE), parse_edit({"op": "remove_node", "node": "ghost"}))


def test_fail_edge_retry_budget_and_failures_dont_crash():
    from gap.graph.interpreter import run_graph
    from gap.sim.task_sampler import sample
    from gap.sim.workcell import Workcell
    g = Graph.model_validate({"task": "place", "nodes": [
        {"id": "find", "skill": "locate_object_gt", "in": {"color": "red"}},
        {"id": "check", "skill": "verify_grasp", "in": {"color": "red"}},  # nothing grasped: fails
    ], "edges": [{"from": "check", "to": "find", "on": "fail", "max_retries": 2}]})
    res = run_graph(g, Workcell(), sample("place", seed=50011))
    assert [t.node_id for t in res.trace] == ["find", "check"] * 3
    assert [t.visit for t in res.trace if t.node_id == "check"] == [0, 1, 2]
    assert res.terminated_by == "node_failure" and not res.success
    assert "rose only" in res.trace[-1].error


def test_unresolvable_input_is_a_node_failure():
    from gap.graph.interpreter import run_graph
    from gap.sim.task_sampler import sample
    from gap.sim.workcell import Workcell
    g = Graph.model_validate({"task": "place", "nodes": [
        {"id": "find", "skill": "locate_object_gt", "in": {"color": "purple"}},
        {"id": "plan", "skill": "plan_top_grasp", "in": {"pose": "find.pose"}},
    ]})
    res = run_graph(g, Workcell(), sample("place", seed=50012))
    assert len(res.trace) == 1 and "purple" in res.trace[0].error


@pytest.mark.slow
def test_place_graph_within_2_points_of_oracle():
    from gap.graph.interpreter import run_graph
    from gap.sim.oracle import run_place_oracle
    from gap.sim.task_sampler import sample
    from gap.sim.workcell import Workcell
    from gap.utils.stats import fmt_rate
    cell, g, seeds = Workcell(), Graph.model_validate(EXAMPLE), range(50000, 50100)
    graph_wins = sum(run_graph(g, cell, sample("place", seed=s)).success for s in seeds)
    oracle_wins = sum(run_place_oracle(cell, sample("place", seed=s)) for s in seeds)
    print("place graph:", fmt_rate(graph_wins, 100), "oracle:", fmt_rate(oracle_wins, 100))
    assert graph_wins >= oracle_wins - 2
