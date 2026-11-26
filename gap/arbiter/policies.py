"""The six arbitration policies compared in Phase 6b.

Each returns a gap.graph.interpreter.ExecutorPolicy for run_graph(). Only nodes whose skill
has a learned executor (fallback.SKILL_TYPE) are arbitrated; every other node runs its skill.
  always_skill        scripted executor only
  always_learned      learned fallback only
  validity_only       skill unless its pre-check fails (AW-0 style)
  fallback_on_failure skill first; learned policy after a failed skill run, continuing from
                      the state the failure left (no restore: what a real robot can do)
  learned_arbiter     executor with the higher predicted success
  arbiter_fallback    learned_arbiter's choice first, the other executor after a failure, as
                      in fallback_on_failure (DECISIONS 2026-11-26, arbiter with a second try)
  oracle              best executor per node in hindsight (upper bound; tries the skill,
                      restores the snapshot if it failed, then runs the learned policy)

The graphs: the reference graphs with RGB-D perception and a verify_grasp width band that
admits every object variant, so a grasp failure is an executor failure, not a band reject.
"""

from __future__ import annotations

import pickle
from functools import cache
from pathlib import Path

import numpy as np

from gap.arbiter.fallback import SKILL_TYPE, mode_of, observe
from gap.graph.edits import SetParam, apply_edit
from gap.graph.schema import Graph
from gap.skills import registry

POLICIES = ("always_skill", "always_learned", "validity_only",
            "fallback_on_failure", "learned_arbiter", "oracle", "arbiter_fallback")
WIDTH_BAND = (0.02, 0.055)   # verify_grasp band covering the 3-5 cm variants


def arbiter_graph(task: str) -> Graph:
    from gap.rehearsal.runner import with_rgbd
    from gap.rehearsal.seeded_bugs import reference_graph
    g = with_rgbd(reference_graph(task))
    for n in g.nodes:
        if n.skill == "verify_grasp":
            g = apply_edit(g, SetParam(node=n.id, param="width_min", value=WIDTH_BAND[0]))
            g = apply_edit(g, SetParam(node=n.id, param="width_max", value=WIDTH_BAND[1]))
    return g


def make_policy(name: str, graph: Graph, arbiter=None, margin: float = 0.0, seed: int = 0,
                obs_mode: str = "privileged"):
    """obs_mode: what the arbiter sees, the learned grasp's own observation."""
    assert name in POLICIES, name
    rng = np.random.default_rng([seed, 23])

    def policy(node_id, cell, instance) -> str:
        node = graph.node(node_id)
        if node.skill not in SKILL_TYPE or name == "always_skill":
            return "skill"
        if name == "always_learned":
            return "learned"
        if name == "validity_only":
            skill = registry.get(node.skill)()
            return "skill" if skill.pre(cell, {}, skill.resolve_params(node.params))[0] else "learned"
        if name == "fallback_on_failure":
            return "fallback"
        if name == "learned_arbiter":
            return arbiter.choose(observe(cell, rng, obs_mode), margin)
        if name == "arbiter_fallback":
            first = arbiter.choose(observe(cell, rng, obs_mode), margin)
            return "fallback" if first == "skill" else "fallback_learned"
        return "oracle"
    return policy


def make_learned_runner(bc_policy, seed: int = 0):
    rng = np.random.default_rng([seed, 29])

    def runner(node, cell, inputs, params):
        return bc_policy.run(cell, params, rng)
    return runner


@cache
def load_models(models_dir: str) -> tuple:
    """(bc grasp policy, arbiter or None) saved by scripts/train_arbiter.py."""
    d = Path(models_dir)
    bc = pickle.loads((d / "bc_grasp.pkl").read_bytes())
    arb = pickle.loads((d / "arbiter.pkl").read_bytes()) if (d / "arbiter.pkl").exists() else None
    return bc, arb


def executors(name: str, models_dir: str, graph: Graph, seed: int):
    """(executor_policy, learned_runner) for run_graph, models loaded once per process.
    name may carry a margin, "learned_arbiter@0.1" (DECISIONS 2026-11-26, time-aware)."""
    bc, arb = load_models(models_dir)
    name, _, margin = name.partition("@")
    return (make_policy(name, graph, arb, float(margin or 0), seed=seed, obs_mode=mode_of(bc)),
            make_learned_runner(bc, seed))
