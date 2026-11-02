"""Static checks before any physics runs.

Agents read these errors and retry, so messages must say what is wrong AND what
would fix it, e.g. "node 'grasp': param approach_height=0.5 outside [0.03, 0.2]".
"""

from __future__ import annotations

from dataclasses import dataclass

from gap.graph.schema import Graph

# Stable error codes. Tests and diagnosis key on these, so do not rename them.
CODES = {
    "empty_graph":     "graph has no nodes",
    "duplicate_id":    "two nodes share an id",
    "unknown_skill":   "node uses a skill not in the registry",
    "unknown_param":   "param not declared by the skill",
    "param_range":     "param outside its declared [lo, hi]",
    "missing_input":   "a required input port is not bound",
    "unknown_input":   "input port not declared by the skill",
    "bad_ref":         "reference points at a missing node or output port",
    "type_mismatch":   "referenced output type differs from the input port type",
    "forward_ref":     "input references a node that has not run yet on every path",
    "bad_edge":        "edge endpoint is not a node",
    "multiple_edges":  "a node has more than one success edge or more than one fail edge",
    "unreachable":     "node cannot be reached from start",
    "bad_start":       "start is not a node id",
}


@dataclass(frozen=True)
class ValidationError:
    code: str
    node: str | None
    message: str

    def __str__(self) -> str:
        where = f"node {self.node!r}: " if self.node else ""
        return f"[{self.code}] {where}{self.message}"


def validate(graph: Graph) -> list[ValidationError]:
    """Return every error found (not just the first). Empty list means valid.

    Hints:
      - Use gap.skills.registry.get(name) for skill metadata.
      - Reachability: build a networkx.DiGraph of success and fail transitions (including the
        implicit next-in-list success edge) and BFS from graph.start_id().
      - forward_ref: a node may only reference nodes that appear on every path before it.
        A simpler, stricter rule that is fine for this project: only reference nodes earlier
        in the `nodes` list.
      - Pydantic already enforces id format and bounded fail edges at parse time.
    """
    raise NotImplementedError


def format_errors(errors: list[ValidationError]) -> str:
    return "\n".join(str(e) for e in errors) if errors else "valid"
