"""Static checks before any physics runs.

Agents read these errors and retry, so messages must say what is wrong AND what
would fix it, e.g. "node 'grasp': param approach_height=0.5 outside [0.03, 0.2]".
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import networkx as nx

from gap.graph.schema import REF_RE, Graph, Node
from gap.skills import registry

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

_PY_TYPES = {"str": (str,), "float": (int, float), "int": (int,), "bool": (bool,)}


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

    forward_ref uses the strict rule: a node may only reference nodes earlier in `nodes`.
    """
    if not graph.nodes:
        return [ValidationError("empty_graph", None, "add at least one node")]
    errs: list[ValidationError] = []
    ids = graph.ids()
    for nid, k in Counter(ids).items():
        if k > 1:
            errs.append(ValidationError("duplicate_id", nid, f"{k} nodes share this id; rename all but one"))
    if graph.start is not None and graph.start not in ids:
        errs.append(ValidationError("bad_start", None, f"start {graph.start!r} is not a node; "
                                                       f"use one of {ids} or omit start"))
    for i, node in enumerate(graph.nodes):
        errs += _check_node(graph, node, ids[:i])
    errs += _check_edges(graph, ids)
    return errs


def format_errors(errors: list[ValidationError]) -> str:
    return "\n".join(str(e) for e in errors) if errors else "valid"


def _check_node(graph: Graph, node: Node, earlier: list[str]) -> list[ValidationError]:
    def err(code: str, msg: str) -> None:
        out.append(ValidationError(code, node.id, msg))

    out: list[ValidationError] = []
    try:
        skill = registry.get(node.skill)
    except KeyError:
        err("unknown_skill", f"skill {node.skill!r} does not exist; choose one of {registry.names()}")
        return out
    for k, v in node.params.items():
        p = skill.params.get(k)
        if p is None:
            err("unknown_param", f"{node.skill} has no param {k!r}; its params are {sorted(skill.params) or 'none'}")
        elif not p.check(v):
            err("param_range", f"param {k}={v} outside [{p.lo}, {p.hi}]; pick a value in range")
    for port in skill.inputs.keys() - node.inputs.keys():
        err("missing_input", f"{node.skill} needs input {port!r} ({skill.inputs[port]}); bind it")
    for port in node.inputs.keys() - skill.inputs.keys():
        err("unknown_input", f"{node.skill} has no input {port!r}; its inputs are {sorted(skill.inputs) or 'none'}")
    for port, value in node.inputs.items():
        if port in skill.inputs:
            out += _check_value(graph, node, port, skill.inputs[port], value, earlier)
    return out


def _check_value(graph, node, port, want, value, earlier) -> list[ValidationError]:
    def err(code: str, msg: str) -> list[ValidationError]:
        return [ValidationError(code, node.id, f"input {port}: {msg}")]

    m = REF_RE.match(value) if isinstance(value, str) else None
    if m and m.group(1) in graph.ids():
        src, out_port = m.groups()
        try:
            src_skill = registry.get(graph.node(src).skill)
        except KeyError:
            return []  # already reported as unknown_skill on the source node
        errs = []
        if out_port not in src_skill.outputs:
            errs += err("bad_ref", f"{src!r} ({src_skill.name}) has no output {out_port!r}; "
                                   f"its outputs are {sorted(src_skill.outputs) or 'none'}")
        elif src_skill.outputs[out_port] != want:
            errs += err("type_mismatch", f"{value} is {src_skill.outputs[out_port]}, port wants {want}")
        if src not in earlier:
            errs += err("forward_ref", f"{src!r} must come before {node.id!r} in the node list")
        return errs
    if want == "Pose":
        if m:
            return err("bad_ref", f"{value!r} names no node in this graph; reference an earlier node")
        return err("type_mismatch", f"Pose inputs must be references like 'find.pose', got {value!r}")
    ok = isinstance(value, _PY_TYPES[want]) and not (want != "bool" and isinstance(value, bool))
    return [] if ok else err("type_mismatch", f"constant {value!r} is not a {want}")


def _check_edges(graph: Graph, ids: list[str]) -> list[ValidationError]:
    errs = []
    for e in graph.edges:
        for end in (e.src, e.to):
            if end not in ids:
                errs.append(ValidationError("bad_edge", None, f"edge {e.src}->{e.to}: {end!r} is not a "
                                                              f"node; use one of {ids}"))
    for (src, on), k in Counter((e.src, e.on) for e in graph.edges).items():
        if k > 1:
            errs.append(ValidationError("multiple_edges", src, f"{k} {on} edges; keep one"))
    if errs or (graph.start is not None and graph.start not in ids):
        return errs  # reachability is meaningless with dangling endpoints
    g = nx.DiGraph()
    g.add_nodes_from(ids)
    for nid in ids:
        nxt, fail = graph.next_on_success(nid), graph.fail_edge(nid)
        if nxt:
            g.add_edge(nid, nxt)
        if fail:
            g.add_edge(nid, fail.to)
    seen = nx.descendants(g, graph.start_id()) | {graph.start_id()}
    for nid in ids:
        if nid not in seen:
            errs.append(ValidationError("unreachable", nid, "no path from start reaches this node; "
                                                            "remove it or add an edge to it"))
    return errs
