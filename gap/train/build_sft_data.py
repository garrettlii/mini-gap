"""SFT data from Phase 5 edit logs.

Output: data/sft/{train,valid}.jsonl in chat format, one example per line:
  {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
mlx-lm-lora renders these with the model's chat template; for Qwen3.5 a finished assistant
turn renders as "<think>\\n\\n</think>\\n\\n{edit}", the same tokens the no-thinking prompt
ends with at run time.

Rules:
  - Keep only accepted edits that improved success (filtered distillation).
  - Build prompts with gap.agents.node_specialist.build_messages, the same function the live
    agent uses, so train and run-time prompts match exactly.
  - Split by task: train on place + stack; sort is held out entirely.
  - Never include anything derived from eval seeds (gap.utils.seeds.is_eval_seed).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from gap.agents.node_specialist import build_messages
from gap.graph.schema import Graph
from gap.rehearsal.diagnose import END
from gap.utils.runs import read_jsonl
from gap.utils.seeds import is_eval_seed, rehearsal_seeds

EDIT_KEYS = ("op", "node", "param", "skill", "value", "rationale")  # the order the prompt shows


def build(edit_logs: list[Path], out_dir: Path, train_tasks=("place", "stack"),
          valid_frac: float = 0.1, seed: int = 0) -> dict[str, int]:
    """Return counts per split."""
    examples = {}
    for log in edit_logs:
        _check_rehearsal_only(Path(log).parent)
        for row in read_jsonl(log):
            if _keep(row, train_tasks):
                ex = to_example(row)
                examples[json.dumps(ex, sort_keys=True)] = ex   # drop exact duplicates
    rows = list(examples.values())
    random.Random(seed).shuffle(rows)
    n_valid = max(1, round(valid_frac * len(rows))) if len(rows) > 1 else 0
    splits = {"valid": rows[:n_valid], "train": rows[n_valid:]}
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        (out_dir / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in part))
    return {name: len(part) for name, part in splits.items()}


def _keep(row: dict, train_tasks) -> bool:
    return (row["task"] in train_tasks and row["node"] != END and row["accepted"]
            and row["success_after"] > row["success_before"]
            and row["edit"]["op"] in ("set_param", "swap_skill"))


def to_example(row: dict) -> dict:
    """One accepted edit row -> one chat example with the specialist's exact prompt."""
    graph = Graph.model_validate_json(row["graph"])
    system, messages = build_messages(graph, row["node"], row["report"])
    edit = {k: row["edit"][k] for k in EDIT_KEYS if k in row["edit"]}
    return {"messages": [{"role": "system", "content": system}, *messages,
                         {"role": "assistant", "content": json.dumps(edit)}]}


def build_grpo(edit_logs: list[Path], out_dir: Path, train_tasks=("place", "stack"),
               n_seeds: int = 8, valid_frac: float = 0.1, seed: int = 0) -> dict[str, int]:
    """GRPO prompts: every (graph, node, report) the specialist was asked about on a train task,
    as {"system", "prompt", "answer": problem key}. Problems the unedited graph already solves
    on all reward seeds are dropped (no reward headroom)."""
    from gap.train.rewards import GRPO_SEED_OFFSET, rehearse_cached, save_problem

    prompts = {}
    for log in edit_logs:
        _check_rehearsal_only(Path(log).parent)
        for row in read_jsonl(log):
            if row["task"] not in train_tasks or row["node"] == END:
                continue
            graph = Graph.model_validate_json(row["graph"])
            system, messages = build_messages(graph, row["node"], row["report"])
            key = save_problem(row["task"], graph, row["node"], out_dir / "problems")
            prompts[(system, key)] = {"system": system, "prompt": messages[0]["content"],
                                      "answer": key, "task": row["task"], "graph": row["graph"]}
    seeds = rehearsal_seeds(n_seeds, GRPO_SEED_OFFSET)
    rows = [{k: v for k, v in p.items() if k not in ("task", "graph")} for p in prompts.values()
            if rehearse_cached(Graph.model_validate_json(p["graph"]), p["task"], seeds)[0] < n_seeds]
    random.Random(seed).shuffle(rows)
    n_valid = max(1, round(valid_frac * len(rows))) if len(rows) > 1 else 0
    splits = {"valid": rows[:n_valid], "train": rows[n_valid:]}
    for name, part in splits.items():
        (out_dir / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in part))
    return {name: len(part) for name, part in splits.items()}


def _check_rehearsal_only(run_dir: Path) -> None:
    """The loop's seeds come from its config; refuse a run that touched eval seeds."""
    cfg = json.loads((run_dir / "config.json").read_text()).get("loop", {})
    seeds = rehearsal_seeds(cfg.get("n_instances", 0), cfg.get("seed_offset", 0))
    if any(is_eval_seed(s) for s in seeds):
        raise ValueError(f"{run_dir} used eval seeds")
