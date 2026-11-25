"""Phase 6 contracts: SFT data building and the GRPO reward.  --phase 6"""

import json

import pytest

from gap.rehearsal.seeded_bugs import buggy_graph

pytestmark = pytest.mark.phase(6)


def _row(task, node, accepted, before, after, edit, graph):
    return {"iter": 0, "task": task, "node": node, "graph": graph.to_json(), "report": "R",
            "edit": edit, "success_before": before, "n": 16, "success_after": after,
            "accepted": accepted, "invalid": None}


def test_sft_keeps_only_improving_specialist_edits_on_train_tasks(tmp_path):
    from gap.agents.node_specialist import build_messages
    from gap.train.build_sft_data import build
    g = buggy_graph("place", "diagonal_grasp")
    fix = {"rationale": "square", "op": "set_param", "node": "plan", "param": "yaw_offset",
           "value": 0.0}
    rows = [_row("place", "plan", True, 0, 16, fix, g),
            _row("place", "plan", False, 0, 0, fix, g),                     # rejected
            _row("sort", "plan_r", True, 0, 16, {**fix, "node": "plan_r"},
                 buggy_graph("sort", "diagonal_grasp")),                      # held-out task
            _row("place", "(end)", True, 0, 16, {"op": "remove_node", "node": "leave"}, g)]
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({"loop": {"n_instances": 16, "seed_offset": 0}}))
    (run / "edits.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    counts = build([run / "edits.jsonl", run / "edits.jsonl"], tmp_path / "sft")
    assert counts == {"valid": 0, "train": 1}, "one example, deduplicated across logs"
    ex = json.loads((tmp_path / "sft" / "train.jsonl").read_text())
    system, messages = build_messages(g, "plan", "R")
    assert ex["messages"][:2] == [{"role": "system", "content": system}, *messages]
    assert json.loads(ex["messages"][2]["content"])["param"] == "yaw_offset"
    assert ex["messages"][2]["content"].startswith('{"op": "set_param"')


def test_reward_table(tmp_path):
    from gap.train.rewards import save_problem, score_edit
    key = save_problem("place", buggy_graph("place", "diagonal_grasp"), "plan", tmp_path)
    score = lambda text: score_edit(key, text, n_seeds=2, problems_dir=tmp_path)
    assert score("I would lower the grasp.") == 0.0
    assert score('{"op": "set_param", "node": "plan", "param": "yaw_offset", "value": 9}') == 0.05
    assert score('{"op": "set_param", "node": "plan", "param": "approach_height", '
                 '"value": 0.1}') == pytest.approx(0.1)
    assert score('{"op": "set_param", "node": "plan", "param": "yaw_offset", '
                 '"value": 0.0}') == pytest.approx(1.1)
