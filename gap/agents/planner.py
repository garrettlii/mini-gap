"""Planner agent: instruction -> first graph.

Sees: instruction, workcell description, skill catalog. Never: success.py, rehearsal code.
"""

from __future__ import annotations

import json
from pathlib import Path

import pydantic

from gap.agents.context import SCHEMA_RULES, WORKCELL, fill
from gap.agents.llm import LLM
from gap.graph.schema import Graph
from gap.graph.validate import format_errors, validate
from gap.skills import catalog_text

PROMPT = (Path(__file__).parent / "prompts" / "planner.md").read_text()
RETRY_TEMPERATURE = 0.5


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
    for attempt in range(max_attempts):
        # Retries sample (seeded, so cached runs repeat): greedy decoding repeats its mistakes.
        text = llm.complete(system, messages, max_tokens=2000,
                            temperature=0.0 if attempt == 0 else RETRY_TEMPERATURE).text
        graph, problem = parse_graph(text)
        log.append(problem or "valid")
        if graph is not None and problem is None:
            return graph, log
        messages += [{"role": "assistant", "content": text},
                     {"role": "user", "content": f"That graph has errors:\n{problem}\n"
                                                 "Fix them and output the whole graph JSON again."}]
    return None, log


def parse_graph(text: str) -> tuple[Graph | None, str | None]:
    """(graph, None) if valid; otherwise (graph or None, short error text for the model)."""
    try:
        raw = first_json_object(text)
    except json.JSONDecodeError as e:
        near = e.doc[max(0, e.pos - 60): e.pos].replace("\n", " ")
        return None, f"invalid JSON ({e.msg}) right after: ...{near}"
    except ValueError as e:
        return None, str(e)
    try:
        graph = Graph.model_validate(_drop_placeholders(raw))
    except pydantic.ValidationError as e:
        return None, _short_errors(e)
    errors = validate(graph)
    return graph, (format_errors(errors) if errors else None)


def first_json_object(text: str) -> dict:
    """The first complete JSON object in text; ignores fences and anything after it."""
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in the output; output one graph object")
    obj, _ = json.JSONDecoder().raw_decode(text, start)
    if not isinstance(obj, dict):
        raise TypeError("output one JSON object")
    return obj


def _drop_placeholders(raw: dict) -> dict:
    """Remove format noise a small model emits for skills without inputs, like "in": {"none": {}}.

    Only values that can never be valid (dicts, lists, null) are dropped; the validator still
    reports any input that is then missing.
    """
    for node in raw.get("nodes", []) if isinstance(raw.get("nodes"), list) else []:
        for key in ("in", "params"):
            if isinstance(node, dict) and isinstance(node.get(key), dict):
                node[key] = {k: v for k, v in node[key].items()
                             if k != "none" and not isinstance(v, (dict, list, type(None)))}
    return raw


def _short_errors(e: pydantic.ValidationError, limit: int = 8) -> str:
    """One readable line per problem; collapses pydantic's per-union-member duplicates."""
    lines: list[str] = []
    for err in e.errors():
        loc = [str(x) for x in err["loc"]]
        if loc and loc[-1] in ("str", "float", "int", "bool"):
            loc, msg = loc[:-1], "must be a string, number, or boolean"
        elif err["type"] == "extra_forbidden" and len(loc) >= 3 and loc[0] == "nodes":
            msg = f"{loc[-1]!r} is not a node field" + (
                "; fail edges go in the top-level \"edges\" list" if loc[-1] in ("on", "max_retries", "to") else "")
            loc = loc[:-1]
        else:
            msg = err["msg"]
        line = f"{'.'.join(loc)}: {msg}"
        if line not in lines:
            lines.append(line)
    more = f"\n(and {len(lines) - limit} more)" if len(lines) > limit else ""
    return "\n".join(lines[:limit]) + more
