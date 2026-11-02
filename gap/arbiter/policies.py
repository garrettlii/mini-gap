"""The six arbitration policies compared in Phase 6b.

Each returns a gap.graph.interpreter.ExecutorPolicy for run_graph().
  always_skill        scripted executor only
  always_learned      learned fallback only
  validity_only       skill unless its pre-check fails (AW-0 style)
  fallback_on_failure skill first; learned policy after a failed post-check (needs restore)
  learned_arbiter     executor with the higher predicted success
  oracle              best executor per node in hindsight (upper bound; uses paired data)
"""

from __future__ import annotations

POLICIES = ("always_skill", "always_learned", "validity_only",
            "fallback_on_failure", "learned_arbiter", "oracle")


def make_policy(name: str, **kwargs):
    raise NotImplementedError
