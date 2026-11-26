"""Paired executor outcomes from identical states.

For each rehearsal instance and each arbitrated node:
  snap = cell.snapshot() before the node
  run the scripted skill  -> post_ok_skill;   cell.restore(snap)
  run the learned policy  -> post_ok_learned; cell.restore(snap)
  record features at snap + both labels
Pairs from the same state remove most confounding between "which executor" and
"which situation", which is what makes the arbiter result credible.

Episodes run the arbiter graphs (policies.arbiter_graph) with the camera noise on; after the
probe the episode continues with the skill, so later grasp nodes (stack, sort) are probed too.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gap.arbiter.fallback import SKILL_TYPE, mode_of, observe
from gap.arbiter.policies import arbiter_graph, load_models
from gap.utils.seeds import rehearsal_seeds

ARBITER_SEED_OFFSET = 10_000   # rehearsal block for arbiter data (loop 0, rewards 20k, BC 30k)


def collect(task: str, n_instances: int, out_path: Path, models_dir: str,
            splits=("nominal", "object"), seed_offset: int = ARBITER_SEED_OFFSET,
            workers: int = 6) -> int:
    """Write rows to a JSONL file; return the row count."""
    from gap.rehearsal.runner import map_jobs
    seeds = rehearsal_seeds(n_instances, seed_offset)
    jobs = [(task, s, splits[i % len(splits)], models_dir) for i, s in enumerate(seeds)]
    rows = [r for rs in map_jobs(_job, jobs, workers) for r in rs]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    return len(rows)


def _job(job) -> list[dict]:
    from gap.graph.interpreter import run_graph, run_skill
    from gap.rehearsal.runner import worker_cell
    from gap.sim.task_sampler import sample
    from gap.skills import registry

    task, seed, split, models_dir = job
    graph, (bc, _) = arbiter_graph(task), load_models(models_dir)
    rng = np.random.default_rng([seed, 31])
    rows: list[dict] = []

    def probe(node_id, cell, instance) -> str:
        node = graph.node(node_id)
        if node.skill not in SKILL_TYPE:
            return "skill"
        skill = registry.get(node.skill)()
        snap, obs = cell.snapshot(), observe(cell, rng, mode_of(bc))
        row = {"task": task, "seed": seed, "split": split, "node": node_id, "obs": obs.tolist(),
               "valid": skill.pre(cell, {}, skill.resolve_params(node.params))[0]}
        for kind in ("skill", "learned"):
            t0 = cell.data.time
            entry = run_skill(node.id, node.skill, cell, {}, node.params, executor=kind,
                              learned=(lambda c, i, p: bc.run(c, p, rng)) if kind == "learned"
                              else None)
            row[f"ok_{kind}"], row[f"t_{kind}"] = entry.post_ok, cell.data.time - t0
            cell.restore(snap)
        rows.append(row)
        return "skill"

    try:
        run_graph(graph, worker_cell(camera_noise=True), sample(task, seed, split),
                  executor_policy=probe)
    except Exception as e:  # noqa: BLE001 -- a broken episode just yields fewer rows
        rows.append({"task": task, "seed": seed, "split": split, "error": repr(e)})
    return rows
