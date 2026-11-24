"""Phase 4 contracts: authoring methods and their executors.  uv run pytest --phase 4

Agents are tested against a scripted fake backend, so no model is needed.
"""

import json

import pytest

from gap.agents.context import EXAMPLE_CODE, EXAMPLE_GRAPH
from gap.agents.llm import LLM, LLMResponse

pytestmark = pytest.mark.phase(4)


class Scripted:
    """A backend that replies with canned texts in order and records the prompts it saw."""

    name = "scripted"

    def __init__(self, *replies):
        self.replies, self.seen = list(replies), []

    def complete(self, system, messages, *, temperature, max_tokens):
        self.seen.append((system, messages))
        return LLMResponse(self.replies.pop(0))


def fake_llm(tmp_path, *replies):
    backend = Scripted(*replies)
    return LLM(backend, cache_dir=tmp_path), backend


def test_planner_feeds_validator_errors_back(tmp_path):
    from gap.agents.planner import plan
    broken = json.loads(EXAMPLE_GRAPH)
    broken["nodes"][1]["skill"] = "teleport"
    llm, backend = fake_llm(tmp_path, json.dumps(broken), "```json\n" + EXAMPLE_GRAPH + "\n```")
    graph, log = plan(llm, "hold the blue cube over zone B")
    assert graph is not None and log[-1] == "valid" and "unknown_skill" in log[0]
    assert "teleport" in backend.seen[1][1][-1]["content"], "errors go back to the model"


def test_planner_gives_up_after_max_attempts(tmp_path):
    from gap.agents.planner import plan
    llm, _ = fake_llm(tmp_path, "no json here", "still none")
    graph, log = plan(llm, "x", max_attempts=2)
    assert graph is None and len(log) == 2


def test_prompts_never_mention_the_success_checker(tmp_path):
    from gap.agents.coder import write_code
    from gap.agents.planner import plan
    llm, backend = fake_llm(tmp_path, EXAMPLE_GRAPH, f"```python\n{EXAMPLE_CODE}\n```")
    plan(llm, "lift the red cube")
    write_code(llm, "lift the red cube")
    for system, messages in backend.seen:
        text = system + json.dumps(messages)
        assert "success.py" not in text and "settled_success" not in text and "min_height" not in text


def test_critic_keeps_only_structural_edits(tmp_path):
    from gap.agents.critic import review
    from gap.graph.schema import Graph
    reply = json.dumps({"edits": [
        {"op": "set_param", "node": "close", "param": "force", "value": 0.9},        # not structural
        {"op": "add_edge", "edge": {"from": "reach", "to": "above", "on": "fail", "max_retries": 1}},
        {"op": "remove_node", "node": "ghost"},                                       # cannot apply
    ]})
    llm, _ = fake_llm(tmp_path, reply)
    graph, kept = review(llm, Graph.model_validate(json.loads(EXAMPLE_GRAPH)))
    assert [e.op for e in kept] == ["add_edge"] and graph.fail_edge("reach").to == "above"


@pytest.mark.parametrize("code,problem", [
    ("def run(:\n  pass", "syntax error"),
    ("import os\ndef run():\n  pass", "imports"),
    ("def go():\n  pass", "run()"),
])
def test_check_code_rejects(code, problem):
    from gap.agents.coder import check_code
    assert problem in check_code(code)


def test_code_policy_runs_and_budgets_hold():
    from gap.graph.code_exec import run_code
    from gap.sim.task_sampler import sample
    from gap.sim.workcell import Workcell
    cell = Workcell()
    loop = run_code("def run():\n    while True:\n        pass", cell, sample("place", 50000),
                    max_lines=5000)
    assert loop.terminated_by == "visit_cap"
    spam = run_code("def run():\n    while True:\n        open_gripper(wait_steps=20)", cell,
                    sample("place", 50000), max_skill_calls=5)
    assert spam.terminated_by == "visit_cap" and len(spam.trace) == 5
    swallow = run_code("def run():\n    try:\n        locate_object_gt(color='purple')\n"
                       "    except Exception:\n        pass", cell, sample("place", 50000))
    assert swallow.terminated_by == "end" and swallow.trace[0].error
    broken = run_code("def run():\n    undefined_skill()", cell, sample("place", 50000))
    assert broken.terminated_by == "exception"


def test_example_code_policy_holds_the_cube():
    from gap.graph.code_exec import run_code
    from gap.sim.task_sampler import sample
    from gap.sim.workcell import Workcell
    res = run_code(EXAMPLE_CODE, Workcell(), sample("sort", 50001))
    assert res.terminated_by == "end" and all(t.post_ok for t in res.trace)
    assert res.trace[-1].state_after["objects"]["blue_cube"].pos[2] > 0.1
