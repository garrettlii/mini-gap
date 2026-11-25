"""Turns per-node traces into failure reports agents can act on.

The useful line is the numeric pattern: "grasp slip in 9 of 32 runs, 8 of them
with object yaw above 40 degrees". That is what separates a report an agent can
fix from "the grasp failed".

FIREWALL: reports describe what nodes did, never how task success is judged.
"""

from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from gap.sim.motion import heading
from gap.types import EpisodeResult

CATEGORIES = ("ik_unreachable", "missed_pose", "grasp_slip", "collision", "timeout",
              "perception_error", "exception", "other")
END = "(end)"   # pseudo-node: every node succeeded but the task was not achieved
ROBOT = ("hand", "left_finger", "right_finger", *(f"link{i}" for i in range(1, 8)))


@dataclass
class NodeFailure:
    node_id: str
    count: int
    total: int
    categories: dict[str, int]
    patterns: list[str] = field(default_factory=list)   # human-readable numeric patterns
    examples: list[str] = field(default_factory=list)   # distinct error messages, most common first
    measured: list[str] = field(default_factory=list)   # geometry at the failing node


@dataclass
class FailureReport:
    n: int
    successes: int
    blamed: list[NodeFailure]           # sorted by count, descending

    def to_text(self, node_id: str | None = None) -> str:
        """Full report, or only one node's section (what the node specialist sees)."""
        lines = [f"Rehearsal: the task was achieved in {self.successes} of {self.n} runs."]
        for f in self.blamed:
            if node_id is None or f.node_id == node_id:
                lines += _section(f)
        return "\n".join(lines)


def _section(f: NodeFailure) -> list[str]:
    if f.node_id == END:
        head = (f"All nodes succeeded but the task was not achieved in {f.count} of {f.total} "
                "runs: a step may be missing or in the wrong order.")
    else:
        cats = ", ".join(f"{c} {k}" for c, k in sorted(f.categories.items(), key=lambda x: -x[1]))
        head = f"Node {f.node_id!r} failed in {f.count} of {f.total} runs ({cats})."
    out = [head]
    out += [f"  error: {e}" for e in f.examples]
    out += [f"  measured: {m}" for m in f.measured]
    out += [f"  pattern: {p}" for p in f.patterns]
    return out


def blame(result: EpisodeResult) -> str | None:
    """First node whose post-check failed (or whose pre-check failed, or that raised)."""
    return next((t.node_id for t in result.trace if t.error), None)


def categorize(result: EpisodeResult, node_id: str) -> str:
    """Map the blamed node's trace entry to one of CATEGORIES using its states and message."""
    if result.terminated_by == "visit_cap":
        return "timeout"
    entry = next((t for t in result.trace if t.node_id == node_id and t.error), None)
    if entry is None:
        return "exception" if result.terminated_by == "exception" else "other"
    err = entry.error or ""
    if "unreachable" in err or "out of reach" in err:
        return "ik_unreachable"
    if err.split(":")[0].endswith(("Error", "Exception")):
        return "exception"
    if entry.skill.startswith("locate_object"):
        return "perception_error"
    if _robot_hit_something(entry.state_after, entry.state_before):
        return "collision"
    if entry.skill in ("close_gripper", "verify_grasp") or "rose only" in err:
        return "grasp_slip"
    if "from target" in err:
        return "missed_pose"
    return "other"


def _robot_hit_something(after: dict, before: dict) -> bool:
    """A new contact between the arm and the table, not present before the node ran."""
    if not after:
        return False
    new = set(map(tuple, after.get("contacts", []))) - set(map(tuple, before.get("contacts", [])))
    return any("world" in pair and set(pair) & set(ROBOT) for pair in new)


def find_patterns(results: list[EpisodeResult], node_id: str, n_bins: int = 3,
                  min_count: int = 3, top: int = 3) -> list[str]:
    """Bin each instance feature (instance.features) and report bins where this node's
    failure rate is much higher than its overall rate. Start simple: quantile bins and a
    minimum count, before anything statistical."""
    failed = np.array([_blamed(r) == node_id for r in results])
    if failed.sum() < min_count:
        return []
    overall = failed.mean()
    keys = sorted({k for r in results for k in r.instance.features if not k.endswith("_raw_yaw")})
    found = []
    for key in keys:
        vals = np.array([r.instance.features.get(key, np.nan) for r in results])
        ok = ~np.isnan(vals)
        if ok.sum() < n_bins * min_count:
            continue
        edges = np.quantile(vals[ok], np.linspace(0, 1, n_bins + 1))
        for lo, hi in itertools.pairwise(edges):
            in_bin = ok & (vals >= lo) & (vals <= hi)
            k, n = int(failed[in_bin].sum()), int(in_bin.sum())
            rate = k / max(n, 1)
            if k >= min_count and rate >= 1.5 * overall and rate - overall >= 0.15:
                text = (f"fails in {k} of {n} runs ({rate:.0%}) when {key} "
                        f"is in [{lo:.2f}, {hi:.2f}], vs {overall:.0%} overall")
                found.append((rate - overall, text))
    return [text for _, text in sorted(found, reverse=True)[:top]]


def grasp_geometry(results: list[EpisodeResult], node_id: str) -> list[str]:
    """How the hand met the nearest cube when node_id failed: its yaw against the cube's faces
    and its offset from the cube's center, averaged over the failures."""
    angles, offsets, heights = [], [], []
    for r in results:
        entry = next((t for t in r.trace if t.node_id == node_id and t.error), None)
        state = entry.state_before if entry else {}
        if not state.get("objects"):
            continue
        tcp = state["tcp"]
        cube = min(state["objects"].values(),
                   key=lambda c: np.hypot(c.pos[0] - tcp.pos[0], c.pos[1] - tcp.pos[1]))
        d = heading(tcp.quat) - heading(cube.quat)
        angles.append(abs(np.degrees((d + np.pi / 4) % (np.pi / 2) - np.pi / 4)))
        offsets.append(np.hypot(cube.pos[0] - tcp.pos[0], cube.pos[1] - tcp.pos[1]))
        heights.append(tcp.pos[2] - cube.pos[2])
    if not angles:
        return []
    text = (f"the hand was {np.mean(angles):.0f} deg off square to the nearest cube's faces "
            f"(0 = fingers flat on two faces); fingertip center {1000 * np.mean(offsets):.0f} mm "
            f"sideways and {1000 * np.mean(heights):+.0f} mm vertically from the cube's center "
            "(averages over these failures)")
    return [text]


def _blamed(result: EpisodeResult) -> str | None:
    if result.success:
        return None
    return blame(result) or END


def diagnose(results: list[EpisodeResult]) -> FailureReport:
    n = len(results)
    per_node: dict[str, list[EpisodeResult]] = {}
    for r in results:
        node = _blamed(r)
        if node is not None:
            per_node.setdefault(node, []).append(r)
    blamed = []
    for node, rs in per_node.items():
        cats = Counter("other" if node == END else categorize(r, node) for r in rs)
        errors = Counter(next((t.error for t in r.trace if t.node_id == node and t.error), "")
                         for r in rs if node != END)
        measured = grasp_geometry(rs, node) if cats.most_common(1)[0][0] == "grasp_slip" else []
        blamed.append(NodeFailure(node, len(rs), n, dict(cats), find_patterns(results, node),
                                  [e for e, _ in errors.most_common(2) if e], measured))
    blamed.sort(key=lambda f: -f.count)
    return FailureReport(n, sum(r.success for r in results), blamed)

