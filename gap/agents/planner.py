"""Planner agent: instruction -> first graph.

Sees: instruction, workcell description, skill catalog. Never: success.py, rehearsal code.
"""

from __future__ import annotations

from pathlib import Path

from gap.agents.llm import LLM
from gap.graph.schema import Graph

PROMPT = (Path(__file__).parent / "prompts" / "planner.md").read_text()


def plan(llm: LLM, instruction: str, max_attempts: int = 3) -> tuple[Graph | None, list[str]]:
    """Return (graph, attempt_log). On validator errors, send them back and retry.

    Fill PROMPT's {catalog}, {workcell}, {instruction}, {schema_rules}. Parse with
    extract_json + Graph.model_validate, then validate(); both kinds of error go back to
    the model verbatim on the next attempt.
    """
    raise NotImplementedError
