"""Structure critic: aggregate failure report -> structural edits.

Handles ordering and missing-step failures: add a retry edge, insert a re-locate or
verify node, remove a redundant node. Emits insert_after / add_edge / remove_node.
"""

from __future__ import annotations

from pathlib import Path

import pydantic

from gap.agents.context import fill
from gap.agents.llm import LLM, extract_json
from gap.graph.edits import AddEdge, EditOp, InsertAfter, RemoveNode, apply_edit, parse_edit
from gap.graph.schema import Graph
from gap.graph.validate import validate
from gap.skills import catalog_text

PROMPT = (Path(__file__).parent / "prompts" / "critic.md").read_text()
STRUCTURAL = (InsertAfter, AddEdge, RemoveNode)
# First-shot review, before any rehearsal: the "multi-agent" planner + critic pair.
STATIC_REVIEW = ("No rehearsal has run yet. Check the graph for missing steps, wrong order, "
                 "or steps that can fail without a recovery path (a fail edge with max_retries).")


def propose_structure(llm: LLM, graph: Graph, report_text: str,
                      max_edits: int = 2) -> list[EditOp]:
    """Up to max_edits structural edits. Unparseable or non-structural edits are dropped."""
    system = fill(PROMPT, graph_json=graph.to_json(), catalog=catalog_text(),
                  report=report_text, max_edits=str(max_edits))
    text = llm.complete(system, [{"role": "user", "content": "Output the edits JSON."}],
                        max_tokens=800).text
    try:
        raw = extract_json(text).get("edits", [])
    except ValueError:
        return []
    edits: list[EditOp] = []
    for item in raw if isinstance(raw, list) else []:
        try:
            edit = parse_edit(item)
        except pydantic.ValidationError:
            continue
        if isinstance(edit, STRUCTURAL):
            edits.append(edit)
    return edits[:max_edits]


def review(llm: LLM, graph: Graph) -> tuple[Graph, list[EditOp]]:
    """First-shot critic pass. Applies edits one by one, keeping each only if the graph
    still validates. Returns (graph, edits kept)."""
    kept: list[EditOp] = []
    for edit in propose_structure(llm, graph, STATIC_REVIEW):
        try:
            candidate = apply_edit(graph, edit)
        except ValueError:
            continue
        if not validate(candidate):
            graph = candidate
            kept.append(edit)
    return graph, kept
