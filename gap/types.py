"""Shared data types. OWNER: done

These are contracts between layers. Changing a field here touches every layer,
so log the reason in docs/DECISIONS.md first.

Conventions
-----------
- Units: meters, radians, seconds, kilograms.
- Quaternions are (w, x, y, z), MuJoCo's convention.
- World frame: robot base at the origin, tabletop at z = 0, +x away from the robot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# The state dict every layer reads. Workcell.get_state() must return exactly these keys:
#   "time":          float, sim seconds
#   "tcp":           Pose of the tool center point (site "tcp")
#   "gripper_width": float, finger opening in meters (0.0 closed, ~0.08 open)
#   "objects":       dict[str, Pose], keyed by body name ("red_cube", ...)
#   "contacts":      list[tuple[str, str]], body-name pairs currently in contact
#   "qpos_arm":      list[float], the 7 arm joint angles
State = dict[str, Any]


@dataclass(frozen=True)
class Pose:
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_arrays(cls, pos: np.ndarray, quat: np.ndarray | None = None) -> Pose:
        q = (1.0, 0.0, 0.0, 0.0) if quat is None else tuple(float(v) for v in quat)
        return cls(tuple(float(v) for v in pos), q)  # type: ignore[arg-type]

    @property
    def p(self) -> np.ndarray:
        return np.asarray(self.pos, dtype=float)

    @property
    def q(self) -> np.ndarray:
        return np.asarray(self.quat, dtype=float)

    def offset(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> Pose:
        return Pose((self.pos[0] + dx, self.pos[1] + dy, self.pos[2] + dz), self.quat)


@dataclass
class TaskInstance:
    """One randomized instance of a task. Fully determined by (task, split, seed)."""

    task: str                      # "lift" | "place" | "stack" | "sort"
    instruction: str               # natural-language command shown to agents
    seed: int
    object_poses: dict[str, Pose]  # initial pose per body name
    split: str = "nominal"
    # Instance variables for diagnosis and the arbiter, e.g. {"red_cube_yaw": 0.7}.
    features: dict[str, float] = field(default_factory=dict)


@dataclass
class SkillResult:
    ok: bool                                   # did run() complete without an internal failure
    outputs: dict[str, Any] = field(default_factory=dict)
    sim_steps: int = 0
    message: str = ""


@dataclass
class TraceEntry:
    """One node execution. The interpreter writes one per node visit."""

    node_id: str
    skill: str
    executor: str                  # "skill" | "learned" (Phase 6b)
    params: dict[str, float]
    inputs: dict[str, Any]
    state_before: State
    state_after: State
    outputs: dict[str, Any]
    pre_ok: bool
    post_ok: bool
    sim_steps: int
    wall_s: float
    visit: int = 0                 # 0 on first visit, increments on retries
    error: str | None = None


@dataclass
class EpisodeResult:
    instance: TaskInstance
    success: bool                  # from gap/sim/success.py, never from agents
    trace: list[TraceEntry]
    sim_time_s: float
    terminated_by: str = "end"     # "end" | "node_failure" | "visit_cap" | "exception"
