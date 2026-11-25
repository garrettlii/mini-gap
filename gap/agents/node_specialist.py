"""Node specialist: one failing node -> one edit to that node.

This is the role Qwen learns in Phase 6, so keep its context small and fixed:
the node, its skill's catalog entry, its trace summary, and the diagnosis lines
for that node. Output is exactly one edit op (set_param or swap_skill).
"""

from __future__ import annotations

from pathlib import Path

import pydantic

from gap.agents.context import fill
from gap.agents.llm import LLM, extract_json
from gap.graph.edits import EditOp, SetParam, SwapSkill, apply_edit, parse_edit
from gap.graph.schema import Graph
from gap.graph.validate import format_errors, validate
from gap.skills import registry

PROMPT = (Path(__file__).parent / "prompts" / "node_specialist.md").read_text()
RETRY_TEMPERATURE = 0.7


def propose_fix(llm: LLM, graph: Graph, node_id: str, node_report: str,
                max_attempts: int = 3, temperature: float = 0.0) -> EditOp | None:
    """One valid edit to node_id, or None. Invalid proposals go back with the reason."""
    system, messages = build_messages(graph, node_id, node_report)
    for attempt in range(max_attempts):
        temp = temperature if attempt == 0 else RETRY_TEMPERATURE
        text = llm.complete(system, messages, max_tokens=200, temperature=temp).text
        edit, problem = check_edit(text, graph, node_id)
        if problem is None:
            return edit
        messages = [*messages, {"role": "assistant", "content": text},
                    {"role": "user", "content": f"That edit is invalid: {problem}\n"
                                                "Output one corrected JSON edit."}]
    return None


def check_edit(text: str, graph: Graph, node_id: str) -> tuple[EditOp | None, str | None]:
    """(edit, None) if text is one valid edit of node_id; else (None, reason for the model)."""
    try:
        edit = parse_edit(extract_json(text))
    except (ValueError, pydantic.ValidationError) as e:
        return None, f"not a valid edit object ({str(e).splitlines()[0]})"
    if not isinstance(edit, (SetParam, SwapSkill)):
        return None, f"op must be set_param or swap_skill, not {edit.op}"
    if edit.node != node_id:
        return None, f"edit node {node_id!r}, not {edit.node!r}"
    errors = validate(apply_edit(graph, edit))
    return (None, format_errors(errors)) if errors else (edit, None)


def build_messages(graph: Graph, node_id: str, node_report: str) -> tuple[str, list[dict]]:
    """Return (system, messages). Shared by the live agent and train/build_sft_data.py, so
    SFT examples see exactly the prompt format the model sees at run time."""
    node = graph.node(node_id)
    skill = registry.get(node.skill)
    swaps = [s.name for s in registry.all_skills().values()
             if s.name != skill.name and s.inputs == skill.inputs and s.outputs == skill.outputs]
    docs = "".join(f"\n    {k}: {prm.doc}" for k, prm in skill.params.items() if prm.doc)
    entry = skill.catalog_entry() + docs + (
        f"\nSkills with the same ports (valid for swap_skill): {', '.join(swaps)}" if swaps
        else "\nNo other skill has the same ports, so use set_param.")
    system = fill(PROMPT, node_json=node.model_dump_json(by_alias=True, exclude_defaults=True),
                  skill_entry=entry, node_report=node_report, node_id=node_id)
    return system, [{"role": "user", "content": f"Output one edit for node {node_id!r}."}]
