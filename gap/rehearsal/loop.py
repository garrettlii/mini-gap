"""Self-learning loop: rehearse -> diagnose -> edit -> rehearse.

Log every step to edits.jsonl as {graph, report, edit, success_before, success_after,
accepted}. These records are the Phase 6 training data, so log rejected edits too.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from gap.graph.edits import EditOp, apply_edit
from gap.graph.schema import Graph
from gap.graph.validate import format_errors, validate
from gap.rehearsal.diagnose import END, FailureReport, diagnose
from gap.rehearsal.runner import rehearse
from gap.sim.task_sampler import sample
from gap.skills import registry
from gap.utils.runs import RunDir
from gap.utils.seeds import rehearsal_seeds

# specialist(graph, node_id, node_report_text) -> one edit or None
Specialist = Callable[[Graph, str, str], EditOp | None]
# critic(graph, full_report_text) -> structural edits
Critic = Callable[[Graph, str], list[EditOp]]
STRUCTURAL = {"timeout", "exception"}   # categories the critic handles, besides END


@dataclass
class LoopConfig:
    task: str
    n_instances: int = 32
    max_iters: int = 12
    patience: int = 3            # stop after this many iterations without improvement
    workers: int = 6
    seed_offset: int = 0         # which block of rehearsal seeds to optimize on
    proposals_per_iter: int = 2  # specialist proposals tried per iteration
    camera_noise: bool = False   # rehearse with the overhead camera noise model


def improve(graph: Graph, cfg: LoopConfig, specialist: Specialist, critic: Critic,
            run: RunDir) -> Graph:
    """Hill-climb on rehearsal seeds only. Keep an edit only if success improves.

    Route by failure category: parameter/skill problems -> specialist for that node;
    ordering/missing-step problems -> critic. Validate every edited graph before rehearsing.
    """
    instances = [sample(cfg.task, s) for s in rehearsal_seeds(cfg.n_instances, cfg.seed_offset)]
    results = rehearse(graph, instances, cfg.workers, keep_states=True,
                        camera_noise=cfg.camera_noise)
    best = sum(r.success for r in results)
    tried: dict[str, list[str]] = {}   # node -> edits already rejected, shown to the specialist
    stale = 0
    for it in range(cfg.max_iters):
        report = diagnose(results)
        if best == cfg.n_instances or not report.blamed:
            break
        improved = False
        for edit, node, text in _proposals(graph, report, specialist, critic, tried, cfg):
            row = {"iter": it, "task": cfg.task, "node": node, "graph": graph.to_json(),
                   "report": text,
                   "edit": edit.model_dump(by_alias=True), "success_before": best,
                   "n": cfg.n_instances, "success_after": None, "accepted": False, "invalid": None}
            candidate = apply_edit(graph, edit)
            errors = validate(candidate)
            if errors:
                run.append_jsonl("edits.jsonl", {**row, "invalid": format_errors(errors)})
                continue
            new_results = rehearse(candidate, instances, cfg.workers, keep_states=True,
                                    camera_noise=cfg.camera_noise)
            after = sum(r.success for r in new_results)
            accepted = after > best
            run.append_jsonl("edits.jsonl", {**row, "success_after": after, "accepted": accepted})
            if accepted:
                graph, results, best, improved = candidate, new_results, after, True
                tried.clear()
                break
        stale = 0 if improved else stale + 1
        if stale >= cfg.patience:
            break
    run.append_jsonl("loop_summary.jsonl", {"task": cfg.task, "final_graph": graph.to_json(),
                                            "success": best, "n": cfg.n_instances})
    return graph


def _proposals(graph: Graph, report: FailureReport, specialist: Specialist, critic: Critic,
               tried: dict[str, list[str]], cfg: LoopConfig):
    """Yield (edit, node, text the agent saw) for the most-blamed failure, routed by category.

    Parameter problems go to the specialist: twice for the failing node, once for each
    upstream suspect. Then the critic always gets one turn on the full report.
    """
    top = report.blamed[0]
    category = max(top.categories, key=top.categories.get)
    if top.node_id != END and category not in STRUCTURAL:
        for i, node in enumerate(suspects(graph, top.node_id, category)):
            text = node_text(report, node)
            for _ in range(cfg.proposals_per_iter if i == 0 else 1):
                seen = text + ("\nAlready tried without improvement: " + "; ".join(tried[node])
                               if tried.get(node) else "")
                edit = specialist(graph, node, seen)
                if edit is None:
                    break
                if _describe(edit) in tried.get(node, []):   # a repeat: don't rehearse it again
                    continue
                tried.setdefault(node, []).append(_describe(edit))
                yield edit, node, seen
    text = report.to_text()
    for edit in critic(graph, text):
        if _describe(edit) not in tried.setdefault(END, []):
            tried[END].append(_describe(edit))
            yield edit, END, text


def node_text(report: FailureReport, node: str) -> str:
    """What the specialist sees about `node`: the most-blamed node's section, plus a pointer
    when `node` is an earlier suspect rather than the node that failed."""
    top = report.blamed[0].node_id
    base = report.to_text(top)
    return base if node == top else (base + f"\nThe failure shows up at node {top!r}; "
                                     f"node {node!r} runs earlier and may cause it.")


def suspects(graph: Graph, node_id: str, category: str, limit: int = 4) -> list[str]:
    """The failing node, then earlier nodes likely to cause it: grasp-pipeline nodes for grasp
    failures, then data-flow ancestors (nodes whose outputs feed it). Only editable nodes."""
    ids = graph.ids()
    earlier = ids[: ids.index(node_id)][::-1]           # nearest first
    out = [node_id]
    if category == "grasp_slip":
        grasp = [n for n in earlier if registry.get(graph.node(n).skill).kind == "grasp"]
        out += grasp
        out += [n for n in earlier if any((graph.parse_ref(v) or ("",))[0] in grasp
                                          for v in graph.node(n).inputs.values())]
    out += _ancestors(graph, node_id)
    editable = [n for n in dict.fromkeys(out) if registry.get(graph.node(n).skill).params]
    return editable[:limit]


def _ancestors(graph: Graph, node_id: str) -> list[str]:
    found: list[str] = []
    stack = [node_id]
    while stack:
        for v in graph.node(stack.pop()).inputs.values():
            ref = graph.parse_ref(v)
            if ref and ref[0] not in found:
                found.append(ref[0])
                stack.append(ref[0])
    return found


def _describe(edit: EditOp) -> str:
    d = edit.model_dump(by_alias=True, exclude={"rationale"})
    return ", ".join(f"{k}={v}" for k, v in d.items())
