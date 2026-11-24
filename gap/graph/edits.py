"""Edit ops: what agents emit after the first graph. Schemas OWNER: done

Edits are cheap to validate, easy to diff, and short enough for a 2B model to learn.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from gap.graph.schema import Edge, Graph, Node


class _Op(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    rationale: str = ""            # one sentence; logged, useful SFT signal


class SetParam(_Op):
    op: Literal["set_param"] = "set_param"
    node: str
    param: str
    value: float


class SwapSkill(_Op):
    op: Literal["swap_skill"] = "swap_skill"
    node: str
    skill: str


class InsertAfter(_Op):
    op: Literal["insert_after"] = "insert_after"
    after: str
    new: Node


class AddEdge(_Op):
    op: Literal["add_edge"] = "add_edge"
    edge: Edge


class RemoveNode(_Op):
    op: Literal["remove_node"] = "remove_node"
    node: str


EditOp = Annotated[Union[SetParam, SwapSkill, InsertAfter, AddEdge, RemoveNode], Field(discriminator="op")]
EDIT_ADAPTER: TypeAdapter = TypeAdapter(EditOp)


def parse_edit(raw: str | dict) -> EditOp:
    """Parse one edit from model output (JSON string or dict). Raises pydantic.ValidationError."""
    return EDIT_ADAPTER.validate_json(raw) if isinstance(raw, str) else EDIT_ADAPTER.validate_python(raw)


def apply_edit(graph: Graph, edit: EditOp) -> Graph:
    """Return a NEW graph with the edit applied; never mutate the input.

    Decide and document: what happens to edges pointing at a removed node? To references
    ("removed.pose") in later nodes' inputs? The validator should catch dangling refs, so
    the simplest correct choice is: remove the node and its edges, leave inputs alone, and
    let validate() report the dangling references back to the agent.

    That is the choice made here. Raises ValueError when the edit names a node that does
    not exist, with a message the agent can act on.
    """
    g = graph.model_copy(deep=True)
    ids = g.ids()
    target = getattr(edit, "node", None) or getattr(edit, "after", None)
    if target is not None and target not in ids:
        raise ValueError(f"{edit.op}: no node {target!r}; nodes are {ids}")
    if isinstance(edit, SetParam):
        g.node(edit.node).params[edit.param] = edit.value
    elif isinstance(edit, SwapSkill):
        g.node(edit.node).skill = edit.skill
    elif isinstance(edit, InsertAfter):
        if edit.new.id in ids:
            raise ValueError(f"insert_after: id {edit.new.id!r} is taken; pick a new id")
        g.nodes.insert(ids.index(edit.after) + 1, edit.new.model_copy(deep=True))
    elif isinstance(edit, AddEdge):
        g.edges.append(edit.edge.model_copy(deep=True))
    elif isinstance(edit, RemoveNode):
        g.nodes = [n for n in g.nodes if n.id != edit.node]
        g.edges = [e for e in g.edges if edit.node not in (e.src, e.to)]
        if g.start == edit.node:
            g.start = None
    return g
