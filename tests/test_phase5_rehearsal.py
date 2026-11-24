"""Phase 5 contracts: rehearsal, diagnosis, specialist parsing, and the loop.  --phase 5"""

import json
from pathlib import Path

import pytest

from gap.graph.schema import Graph
from gap.types import EpisodeResult, Pose, TaskInstance, TraceEntry

pytestmark = pytest.mark.phase(5)

PLACE = Graph.model_validate(json.loads(Path("graphs/place_example.json").read_text()))


def episode(seed, yaw, fail_node=None, error="post-check failed: no cube is touched by both fingers"):
    inst = TaskInstance("place", "x", seed, {"red_cube": Pose((0.5, 0, 0.02))},
                        features={"red_cube_yaw": yaw})
    nodes = ["find", "plan", "close"]
    trace = [TraceEntry(n, "close_gripper" if n == "close" else "locate_object_gt", "skill", {},
                        {}, {"contacts": []}, {"contacts": []}, {}, True, n != fail_node, 0, 0.0,
                        0, error if n == fail_node else None)
             for n in nodes[: nodes.index(fail_node) + 1 if fail_node else 3]]
    return EpisodeResult(inst, fail_node is None, trace, 1.0,
                         "node_failure" if fail_node else "end")


def test_diagnose_blames_counts_and_finds_the_yaw_pattern():
    from gap.rehearsal.diagnose import diagnose
    results = [episode(i, yaw=i / 30, fail_node="close" if i >= 20 else None) for i in range(30)]
    rep = diagnose(results)
    assert rep.successes == 20 and rep.blamed[0].node_id == "close" and rep.blamed[0].count == 10
    assert rep.blamed[0].categories == {"grasp_slip": 10}
    assert any("red_cube_yaw" in p for p in rep.blamed[0].patterns)
    text = rep.to_text("close")
    assert "10 of 30" in text and "both fingers" in text


def test_task_not_achieved_after_all_nodes_is_reported_as_end():
    from gap.rehearsal.diagnose import END, diagnose
    r = episode(1, 0.0)
    r.success = False
    rep = diagnose([r, episode(2, 0.0)])
    assert rep.blamed[0].node_id == END and "missing" in rep.to_text()


def test_reports_never_mention_the_success_checker():
    from gap.rehearsal.diagnose import diagnose
    text = diagnose([episode(i, i / 10, "close") for i in range(10)]).to_text()
    assert "success.py" not in text and "min_height" not in text and "settled" not in text


def test_rehearse_refuses_eval_seeds():
    from gap.rehearsal.runner import rehearse
    from gap.sim.task_sampler import sample
    with pytest.raises(ValueError, match="eval seeds"):
        rehearse(PLACE, [sample("place", 100000)], workers=1)


def test_parallel_rehearsal_matches_inline_and_drops_states():
    from gap.rehearsal.runner import rehearse
    from gap.sim.task_sampler import sample
    insts = [sample("place", s) for s in range(4)]
    inline = rehearse(PLACE, insts, workers=1)
    pooled = rehearse(PLACE, insts, workers=2, keep_states=False)
    assert [r.success for r in inline] == [r.success for r in pooled]
    assert [r.instance.seed for r in pooled] == [0, 1, 2, 3]
    assert all(t.state_before == {} for r in pooled for t in r.trace)


def test_specialist_edit_checks(tmp_path):
    from gap.agents.node_specialist import build_messages, check_edit
    ok, problem = check_edit('{"op": "set_param", "node": "close", "param": "force", "value": 0.9}',
                             PLACE, "close")
    assert problem is None and ok.value == 0.9
    assert "outside" in check_edit('{"op": "set_param", "node": "close", "param": "force", '
                                   '"value": 5}', PLACE, "close")[1]
    assert "not 'reach'" in check_edit('{"op": "set_param", "node": "reach", "param": "speed", '
                                       '"value": 0.2}', PLACE, "close")[1]
    system, _ = build_messages(PLACE, "find", "Node 'find' failed in 3 of 32 runs.")
    assert "locate_object_rgbd" in system, "swap candidates with the same ports are listed"


def test_loop_keeps_the_fixing_edit_and_logs_every_proposal(tmp_path):
    from gap.graph.edits import SetParam
    from gap.rehearsal.loop import LoopConfig, improve
    from gap.rehearsal.seeded_bugs import buggy_graph
    from gap.utils.runs import RunDir, read_jsonl

    asked = []

    def specialist(graph, node, text):     # fixes the depth bug, but only when asked about plan
        asked.append(node)
        return SetParam(node="plan", param="depth", value=0.0) if node == "plan" else None

    run = RunDir("loop_test", {}, root=tmp_path)
    cfg = LoopConfig(task="place", n_instances=3, max_iters=3, workers=1)
    fixed = improve(buggy_graph("place", "grasp_too_deep"), cfg, specialist, lambda g, t: [], run)
    assert fixed.node("plan").params["depth"] == 0.0
    assert asked[0] == "close" and "plan" in asked, "upstream suspects are asked after the blamed node"
    rows = read_jsonl(run.path / "edits.jsonl")
    assert rows[-1]["accepted"] and rows[-1]["success_after"] > rows[-1]["success_before"]
    assert "close" in rows[-1]["report"], "the logged report is the text the specialist saw"
