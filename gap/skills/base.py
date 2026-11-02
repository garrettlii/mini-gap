"""The skill contract. OWNER: done

A skill declares its interface (ports, params, checks) so agents can wire skills
into graphs from the catalog alone, without reading code. The `post` check is
what makes node-level diagnosis possible: without it you only know the whole
graph failed.

Port types are strings so they appear verbatim in prompts and the validator can
compare them: one of PORT_TYPES.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from gap.types import SkillResult

if TYPE_CHECKING:
    from gap.sim.workcell import Workcell

PORT_TYPES = frozenset({"Pose", "str", "float", "int", "bool"})


@dataclass(frozen=True)
class Param:
    default: float
    lo: float
    hi: float
    doc: str = ""

    def check(self, value: float) -> bool:
        return self.lo <= value <= self.hi


class Skill(ABC):
    name: ClassVar[str]
    kind: ClassVar[str]                      # "perception" | "motion" | "grasp" | "planning" | "check"
    description: ClassVar[str]               # one line; goes into prompts
    inputs: ClassVar[dict[str, str]] = {}    # port name -> port type
    outputs: ClassVar[dict[str, str]] = {}
    params: ClassVar[dict[str, Param]] = {}

    def resolve_params(self, overrides: dict[str, float] | None = None) -> dict[str, float]:
        """Defaults merged with overrides. Raises on unknown or out-of-range params."""
        out = {k: p.default for k, p in self.params.items()}
        for k, v in (overrides or {}).items():
            if k not in self.params:
                raise KeyError(f"{self.name}: unknown param {k!r}")
            if not self.params[k].check(v):
                p = self.params[k]
                raise ValueError(f"{self.name}.{k}={v} outside [{p.lo}, {p.hi}]")
            out[k] = float(v)
        return out

    def pre(self, cell: Workcell, inputs: dict[str, Any], params: dict[str, float]) -> tuple[bool, str]:
        """Can this run now? Return (ok, reason). Default: always ok."""
        return True, ""

    def pre_margins(self, cell: Workcell, inputs: dict[str, Any], params: dict[str, float]) -> dict[str, float]:
        """Continuous versions of the pre-check (e.g. IK residual, reach distance).

        Feature input for the Phase 6b arbiter. Default: none.
        """
        return {}

    @abstractmethod
    def run(self, cell: Workcell, inputs: dict[str, Any], params: dict[str, float]) -> SkillResult:
        ...

    def post(self, cell: Workcell, inputs: dict[str, Any], outputs: dict[str, Any],
             params: dict[str, float]) -> tuple[bool, str]:
        """Did it do its job? Return (ok, reason). Every concrete skill must override this."""
        raise NotImplementedError(f"{self.name} has no post-check")

    @classmethod
    def catalog_entry(cls) -> str:
        ins = ", ".join(f"{k}: {t}" for k, t in cls.inputs.items()) or "none"
        outs = ", ".join(f"{k}: {t}" for k, t in cls.outputs.items()) or "none"
        ps = "; ".join(f"{k}={p.default} in [{p.lo}, {p.hi}]" for k, p in cls.params.items()) or "none"
        return f"- {cls.name} ({cls.kind}): {cls.description}\n    in: {ins} | out: {outs} | params: {ps}"
