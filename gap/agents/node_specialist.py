"""Node specialist: one failing node -> one edit to that node.

This is the role Qwen learns in Phase 6, so keep its context small and fixed:
the node, its skill's catalog entry, its trace summary, and the diagnosis lines
for that node. Output is exactly one edit op (set_param or swap_skill).
"""

from __future__ import annotations

from pathlib import Path

from gap.agents.llm import LLM
from gap.graph.edits import EditOp
from gap.graph.schema import Graph

PROMPT = (Path(__file__).parent / "prompts" / "node_specialist.md").read_text()


def propose_fix(llm: LLM, graph: Graph, node_id: str, node_report: str,
                max_attempts: int = 3) -> EditOp | None:
    raise NotImplementedError


def build_messages(graph: Graph, node_id: str, node_report: str) -> tuple[str, list[dict]]:
    """Return (system, messages). Shared by the live agent and train/build_sft_data.py, so
    SFT examples see exactly the prompt format the model sees at run time."""
    raise NotImplementedError
