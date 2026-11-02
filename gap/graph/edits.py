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
    """
    raise NotImplementedError
