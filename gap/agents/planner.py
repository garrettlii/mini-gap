"""Planner agent: instruction -> first graph.

Sees: instruction, workcell description, skill catalog. Never: success.py, rehearsal code.
"""

from __future__ import annotations

from pathlib import Path

import pydantic

from gap.agents.context import SCHEMA_RULES, WORKCELL, fill
from gap.agents.llm import LLM, extract_json
from gap.graph.schema import Graph
from gap.graph.validate import format_errors, validate
from gap.skills import catalog_text

PROMPT = (Path(__file__).parent / "prompts" / "planner.md").read_text()


def plan(llm: LLM, instruction: str, max_attempts: int = 3) -> tuple[Graph | None, list[str]]:
    """Return (graph, attempt_log). On validator errors, send them back and retry.

    Fill PROMPT's {catalog}, {workcell}, {instruction}, {schema_rules}. Parse with
    extract_json + Graph.model_validate, then validate(); both kinds of error go back to
    the model verbatim on the next attempt.
    """
    system = fill(PROMPT, catalog=catalog_text(), workcell=WORKCELL, instruction=instruction,
                  schema_rules=SCHEMA_RULES)
    messages = [{"role": "user", "content": f"Task: {instruction}\nOutput the graph JSON."}]
    log: list[str] = []
    for _ in range(max_attempts):
        text = llm.complete(system, messages, max_tokens=2000).text
        graph, problem = parse_graph(text)
        log.append(problem or "valid")
        if graph is not None and problem is None:
            return graph, log
        messages += [{"role": "assistant", "content": text},
                     {"role": "user", "content": f"That graph has errors:\n{problem}\n"
                                                 "Fix them and output the whole graph JSON again."}]
    return None, log


def parse_graph(text: str) -> tuple[Graph | None, str | None]:
    """(graph, None) if valid; otherwise (graph or None, error text for the model)."""
    try:
        graph = Graph.model_validate(extract_json(text))
    except (ValueError, pydantic.ValidationError) as e:
        return None, f"not a valid graph object: {e}"
    errors = validate(graph)
    return graph, (format_errors(errors) if errors else None)
