"""Structure critic: aggregate failure report -> structural edits.

Handles ordering and missing-step failures: add a retry edge, insert a re-locate or
verify node, remove a redundant node. Emits insert_after / add_edge / remove_node.
"""

from __future__ import annotations

from pathlib import Path

from gap.agents.llm import LLM
from gap.graph.edits import EditOp
from gap.graph.schema import Graph

PROMPT = (Path(__file__).parent / "prompts" / "critic.md").read_text()


def propose_structure(llm: LLM, graph: Graph, report_text: str,
                      max_edits: int = 2) -> list[EditOp]:
    raise NotImplementedError
