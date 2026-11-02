"""Graph spec: the policy format agents write and Qwen learns to emit. OWNER: done

Semantics (keep these stable; prompts and training data depend on them):
  - Execution starts at `start`, or the first node if omitted.
  - On success, control goes to the target of the node's explicit success edge if it has
    one, otherwise to the next node in the `nodes` list. The last node ends the episode.
  - On failure (pre or post check false, or exception), control follows the node's fail
    edge if it has one and that edge's retry budget is not used up; otherwise the episode
    ends with terminated_by="node_failure".
  - An input value is a reference if it is a string "<node_id>.<output_port>" whose node_id
    is a node in this graph. Every other value is a constant.

Example: graphs/place_example.json
"""

from __future__ import annotations

import re
from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

ID_RE = r"^[a-z][a-z0-9_]{0,31}$"
REF_RE = re.compile(r"^([a-z][a-z0-9_]*)\.([a-z][a-z0-9_]*)$")

InputValue = Union[str, float, int, bool]


class Node(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    id: str = Field(pattern=ID_RE)
    skill: str
    inputs: dict[str, InputValue] = Field(default_factory=dict, alias="in")
    params: dict[str, float] = Field(default_factory=dict)


class Edge(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    src: str = Field(alias="from")
    to: str
    on: Literal["success", "fail"] = "success"
    max_retries: int | None = Field(default=None, ge=1, le=5)

    @model_validator(mode="after")
    def _fail_edges_are_bounded(self) -> Edge:
        if self.on == "fail" and self.max_retries is None:
            raise ValueError(f"fail edge {self.src}->{self.to} needs max_retries")
        if self.on == "success" and self.max_retries is not None:
            raise ValueError(f"success edge {self.src}->{self.to} must not set max_retries")
        return self


class Graph(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str
    start: str | None = None
    nodes: list[Node]
    edges: list[Edge] = Field(default_factory=list)

    def node(self, node_id: str) -> Node:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(node_id)

    def ids(self) -> list[str]:
        return [n.id for n in self.nodes]

    def start_id(self) -> str:
        return self.start or self.nodes[0].id

    def parse_ref(self, value: InputValue) -> tuple[str, str] | None:
        """('find', 'pose') for 'find.pose' if 'find' is a node here, else None."""
        if isinstance(value, str):
            m = REF_RE.match(value)
            if m and m.group(1) in self.ids():
                return m.group(1), m.group(2)
        return None

    def next_on_success(self, node_id: str) -> str | None:
        for e in self.edges:
            if e.src == node_id and e.on == "success":
                return e.to
        ids = self.ids()
        i = ids.index(node_id)
        return ids[i + 1] if i + 1 < len(ids) else None

    def fail_edge(self, node_id: str) -> Edge | None:
        for e in self.edges:
            if e.src == node_id and e.on == "fail":
                return e
        return None

    def to_json(self) -> str:
        return self.model_dump_json(by_alias=True, exclude_none=True, exclude_defaults=True)
