"""Runs a free-form code policy (the code-as-policies baseline) on a Workcell.

Same skills, same pre/post checks, same success check as graphs: only the structure differs,
which is what the "graph structure matters" comparison needs. Skills are plain functions;
each returns its outputs dict or raises SkillFailed.

Not a security sandbox: the code comes from the local model and runs with a small builtin
whitelist (no imports), a cap on skill calls, and a cap on executed policy lines.
"""

from __future__ import annotations

import builtins
import sys
from collections import Counter

from gap.graph.interpreter import run_skill
from gap.sim.success import settled_success
from gap.sim.workcell import Workcell
from gap.skills import registry
from gap.types import EpisodeResult, TaskInstance, TraceEntry

FILENAME = "<policy>"
SAFE_BUILTINS = {n: getattr(builtins, n) for n in (
    "range", "len", "min", "max", "abs", "round", "float", "int", "str", "bool", "list", "dict",
    "tuple", "enumerate", "zip", "isinstance", "print", "Exception", "ValueError", "KeyError")}


class SkillFailed(Exception):
    """Raised into policy code when a skill's pre-check, run, or post-check fails."""


class BudgetExceeded(BaseException):  # BaseException: policy code's `except Exception` can't eat it
    pass


def run_code(code: str, cell: Workcell, instance: TaskInstance, *, max_skill_calls: int = 60,
             max_lines: int = 100_000, frame_sink: list | None = None) -> EpisodeResult:
    cell.frame_sink = frame_sink
    cell.reset(instance)
    trace: list[TraceEntry] = []
    env = {"__builtins__": SAFE_BUILTINS, "SkillFailed": SkillFailed}
    env.update({name: _skill_fn(name, cell, trace, max_skill_calls) for name in registry.names()})
    terminated = "end"
    lines = 0

    def tracer(frame, event, arg):
        nonlocal lines
        if frame.f_code.co_filename != FILENAME:
            return None  # only count the policy's own lines, not the skills'
        lines += 1
        if lines > max_lines:
            raise BudgetExceeded(f"policy ran more than {max_lines} lines")
        return tracer

    sys.settrace(tracer)
    try:
        exec(compile(code, FILENAME, "exec"), env)  # noqa: S102 -- the baseline under test
        env["run"]()
    except SkillFailed:
        terminated = "node_failure"
    except BudgetExceeded:
        terminated = "visit_cap"
    except Exception:  # noqa: BLE001 -- broken policy code ends the episode, not the run
        terminated = "exception"
    finally:
        sys.settrace(None)
    success = settled_success(cell)
    cell.frame_sink = None
    return EpisodeResult(instance, success, trace, float(cell.data.time), terminated)


def _skill_fn(name: str, cell: Workcell, trace: list[TraceEntry], max_calls: int):
    calls = Counter()

    def call(**kwargs):
        if len(trace) >= max_calls:
            raise BudgetExceeded(f"more than {max_calls} skill calls")
        ports = registry.get(name).inputs
        inputs = {k: v for k, v in kwargs.items() if k in ports}
        params = {k: v for k, v in kwargs.items() if k not in ports}
        missing = ports.keys() - inputs.keys()
        error = f"missing input(s) {sorted(missing)}" if missing else None
        entry = run_skill(f"{name}_{calls[name]}", name, cell, inputs, params, error=error)
        calls[name] += 1
        trace.append(entry)
        if not entry.post_ok:
            raise SkillFailed(entry.error)
        return entry.outputs

    call.__name__ = name
    return call
