"""GRPO reward: validate + rehearse the edited graph.

scripts/train_grpo.py registers a batched wrapper of score_edit with mlx-lm-lora. The reward
only sees prompts, completions, and the dataset's answer field, so `answer` holds a problem
key ("place/ab12cd34ef56") naming data/grpo/problems/place/ab12cd34ef56.json, which stores the
graph, the node to edit, and the task.

Reward table (from the build guide):
    not parseable JSON                  0.0
    parses but fails validation         0.05
    valid                               0.1 + (success_after - success_before) / n_seeds
    valid but cycle time grows > 50%    subtract 0.2

Cycle time is the mean sim time of successful episodes (failed episodes end early, so an edit
that fixes them would otherwise look slower). Rehearsals are cached by the (edited) graph:
GRPO samples the same edit often. Read 20 sampled edits by hand after every run to catch reward
hacking (deleting verify_grasp, maxing retries, inflating wait_steps).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pydantic

from gap.agents.llm import extract_json
from gap.agents.node_specialist import check_edit
from gap.graph.edits import apply_edit, parse_edit
from gap.graph.schema import Graph
from gap.rehearsal.runner import rehearse
from gap.sim.task_sampler import sample
from gap.utils.seeds import rehearsal_seeds

PROBLEMS = Path(__file__).resolve().parents[2] / "data" / "grpo" / "problems"
GRPO_SEED_OFFSET = 20_000   # rehearsal block for rewards; the loop uses 0, held-out 40,000
_cache: dict[tuple[str, str], tuple[int, float | None]] = {}


def score_edit(answer_key: str, completion: str, n_seeds: int = 8,
               problems_dir: Path = PROBLEMS) -> float:
    """Pure function the registered reward wraps. Unit-test this directly."""
    prob = json.loads((problems_dir / f"{answer_key}.json").read_text())
    graph = Graph.model_validate_json(prob["graph"])
    try:
        parse_edit(extract_json(completion))
    except (ValueError, pydantic.ValidationError):
        return 0.0
    edit, problem = check_edit(completion, graph, prob["node"])
    if problem is not None:
        return 0.05
    seeds = rehearsal_seeds(n_seeds, GRPO_SEED_OFFSET)
    before, t_before = rehearse_cached(graph, prob["task"], seeds)
    after, t_after = rehearse_cached(apply_edit(graph, edit), prob["task"], seeds)
    slower = t_before is not None and t_after is not None and t_after > 1.5 * t_before
    return 0.1 + (after - before) / n_seeds - (0.2 if slower else 0.0)


def rehearse_cached(graph: Graph, task: str, seeds: list[int]) -> tuple[int, float | None]:
    """(successes, mean sim time of the successful episodes or None), cached."""
    key = (graph.to_json() + task, json.dumps(seeds))
    if key not in _cache:
        results = rehearse(graph, [sample(task, s) for s in seeds], keep_states=False)
        times = [r.sim_time_s for r in results if r.success]
        _cache[key] = (len(times), float(np.mean(times)) if times else None)
    return _cache[key]


def problem_key(task: str, graph: Graph, node: str) -> str:
    return f"{task}/{hashlib.sha1((graph.to_json() + node).encode()).hexdigest()[:12]}"


def save_problem(task: str, graph: Graph, node: str, problems_dir: Path = PROBLEMS) -> str:
    key = problem_key(task, graph, node)
    path = problems_dir / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"task": task, "node": node, "graph": graph.to_json()}))
    return key
