"""Turns per-node traces into failure reports agents can act on.

The useful line is the numeric pattern: "grasp slip in 9 of 32 runs, 8 of them
with object yaw above 40 degrees". That is what separates a report an agent can
fix from "the grasp failed".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from gap.types import EpisodeResult

CATEGORIES = ("ik_unreachable", "missed_pose", "grasp_slip", "collision", "timeout",
              "perception_error", "exception", "other")


@dataclass
class NodeFailure:
    node_id: str
    count: int
    total: int
    categories: dict[str, int]
    patterns: list[str] = field(default_factory=list)   # human-readable numeric patterns


@dataclass
class FailureReport:
    n: int
    successes: int
    blamed: list[NodeFailure]           # sorted by count, descending

    def to_text(self, node_id: str | None = None) -> str:
        """Full report, or only one node's section (what the node specialist sees)."""
        raise NotImplementedError


def blame(result: EpisodeResult) -> str | None:
    """First node whose post-check failed (or whose pre-check failed, or that raised)."""
    raise NotImplementedError


def categorize(result: EpisodeResult, node_id: str) -> str:
    """Map the blamed node's trace entry to one of CATEGORIES using its states and message."""
    raise NotImplementedError


def find_patterns(results: list[EpisodeResult], node_id: str, n_bins: int = 3) -> list[str]:
    """Bin each instance feature (instance.features) and report bins where this node's
    failure rate is much higher than its overall rate. Start simple: quantile bins and a
    minimum count, before anything statistical."""
    raise NotImplementedError


def diagnose(results: list[EpisodeResult]) -> FailureReport:
    raise NotImplementedError
