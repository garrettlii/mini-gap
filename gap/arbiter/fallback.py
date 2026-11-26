"""Learned fallback executors, one per manipulation node type.

Implemented: grasp (close_gripper nodes). A closed-loop MLP (sklearn), behavior-cloned from a
privileged servo demonstrator on rehearsal seeds. Two observation modes: "privileged" sees the
nearest object from the hand (gap.skills._common.hand_offsets plus small noise; a stand-in for
a wrist camera), "wrist" has to find it in the wrist RGB-D image (gap.arbiter.wrist). It
outputs a hand-frame velocity (closing axis, width axis, up, yaw) and a close
command. A damped-least-squares Jacobian step turns the velocity into joint targets, the way
end-effector-delta policies run on real arms. Once it commands close (or its time runs out)
it closes like close_gripper. The node's post-check judges it like the skill.

Not yet: reach and place types (their post-checks compare against the scripted target pose).
"""

from __future__ import annotations

import mujoco
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gap.arbiter.wrist import observe_wrist
from gap.sim import model_builder as mb
from gap.sim.motion import heading, move_tcp, set_gripper
from gap.skills._common import hand_offsets
from gap.skills.grasp import CUBE_WIDTH_CTRL
from gap.types import Pose, SkillResult

NODE_TYPES = ("reach", "grasp", "place")
SKILL_TYPE = {"close_gripper": "grasp"}     # node skills with a learned executor

CTRL_STEPS = 10                             # sim steps per control step (20 ms)
MAX_CTRL = 150                              # 3 s to line up before closing anyway
V_XY, V_Z, W_YAW = 0.06, 0.04, 0.8          # demonstrator speed limits: m/s, m/s, rad/s
GAIN = 4.0                                  # demonstrator proportional gain, 1/s
MIN_TCP_Z = 0.014                           # keep the fingertips off the table
OBS_SD = np.array([0.001, 0.001, 0.001, np.deg2rad(1.0)])  # wrist-view noise
OBS_KEYS = ("close_axis", "width_axis", "dz", "yaw", "half", "round")
ACT_SCALE = np.array([V_XY, V_XY, V_Z, W_YAW, 1.0])


def observe(cell, rng: np.random.Generator, mode: str = "privileged") -> np.ndarray:
    """[close_axis, width_axis, dz, yaw, half, round, gripper width, tcp height]; the wrist
    mode appends `seen` (gap.arbiter.wrist)."""
    if mode == "wrist":
        return observe_wrist(cell, rng)
    off = hand_offsets(cell)
    x = np.array([off[k] for k in OBS_KEYS], dtype=float)
    x[:4] += rng.normal(0.0, OBS_SD)
    state = cell.get_state()
    return np.r_[x, state["gripper_width"], state["tcp"].pos[2]]


def demonstrator(obs: np.ndarray) -> np.ndarray:
    """Privileged servo: line the fingers up on the object's center, then close.
    Returns [v_close, v_width, v_up, w_yaw, close] scaled to [-1, 1] by ACT_SCALE."""
    ca, wa, dz, yaw, _, _, _, tcp_z = obs
    ez = max(tcp_z + dz, MIN_TCP_Z) - tcp_z
    v = np.clip(GAIN * np.array([ca, wa, ez, yaw]), -ACT_SCALE[:4], ACT_SCALE[:4])
    aligned = abs(ca) < 0.002 and abs(wa) < 0.004 and abs(ez) < 0.003 and abs(yaw) < 0.05
    return np.r_[v, float(aligned)] / ACT_SCALE


class FallbackPolicy:
    def __init__(self, node_type: str = "grasp", obs_mode: str = "privileged"):
        assert node_type in NODE_TYPES
        self.node_type = node_type
        self.obs_mode = obs_mode
        self.model = make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(64, 64), max_iter=400, early_stopping=True, random_state=0))

    def fit(self, X: np.ndarray, Y: np.ndarray, close_weight: int = 10) -> FallbackPolicy:
        """Close commands are rare (one per demo), so they are repeated close_weight times."""
        close = Y[:, 4] > 0.5
        reps = np.where(close, close_weight, 1)
        self.model.fit(np.repeat(X, reps, 0), np.repeat(Y, reps, 0))
        return self

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self.model.predict(obs[None])[0]

    def run(self, cell, params: dict, rng: np.random.Generator) -> SkillResult:
        """The learned grasp: servo with the MLP until it says close, then close."""
        return servo_and_close(cell, params, self.act, rng, mode=mode_of(self))


def mode_of(policy) -> str:
    return getattr(policy, "obs_mode", "privileged")   # policies pickled before wrist mode


def servo_and_close(cell, params: dict, policy, rng: np.random.Generator,
                    record: list | None = None, act_noise: float = 0.0,
                    mode: str = "privileged") -> SkillResult:
    """Closed loop: observe -> policy -> joint targets, until close > 0.5 or MAX_CTRL.
    policy=None runs the demonstrator on the privileged view; record (demos) collects
    (observation in `mode`, demonstrator action); act_noise perturbs the executed action."""
    ctrl = cell.data.ctrl.copy()
    ctrl[7] = mb.GRIPPER_OPEN_CTRL
    for _ in range(MAX_CTRL):
        obs = observe(cell, rng, mode)
        teacher = obs if mode == "privileged" else observe(cell, rng)
        a = demonstrator(teacher) if policy is None else policy(obs)
        if record is not None:
            record.append((obs, demonstrator(teacher)))
        if a[4] > 0.5:
            break
        a = a + rng.normal(0.0, act_noise, 5) if act_noise else a
        ctrl[:7] += _joint_step(cell, np.clip(a[:4], -1, 1) * ACT_SCALE[:4])
        cell.step(ctrl, CTRL_STEPS)
    ctrl[7] = CUBE_WIDTH_CTRL * (1 - params["force"])
    cell.step(ctrl, int(params["wait_steps"]))
    return SkillResult(True)


def _joint_step(cell, v: np.ndarray) -> np.ndarray:
    """Hand-frame velocity [v_close, v_width, v_up, w_yaw] -> joint target change for one
    control step (damped least squares on the TCP Jacobian; roll and pitch held at zero)."""
    m, d = cell.model, cell.data
    cdir = heading(cell.get_state()["tcp"].quat) - np.pi / 2
    c, s = np.cos(cdir), np.sin(cdir)
    twist = np.array([v[0] * c - v[1] * s, v[0] * s + v[1] * c, v[2], 0.0, 0.0, v[3]])
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jacp, jacr, m.site(mb.TCP_SITE).id)
    dofs = [m.jnt_dofadr[m.joint(n).id] for n in mb.ARM_JOINTS]
    J = np.vstack([jacp, jacr])[:, dofs]
    dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), twist)
    return dq * CTRL_STEPS * m.opt.timestep


def demo_start(cell, cube: str, rng: np.random.Generator) -> bool:
    """Put the open hand at a perturbed grasp pose around `cube` (the kind of pose a noisy
    perception estimate leads to). False if the arm could not get there."""
    obj = cell.get_state()["objects"][cube]
    yaw = heading(obj.quat) + rng.uniform(-0.35, 0.35)
    off = rng.uniform(-0.02, 0.02, 2)
    z = max(obj.pos[2] + rng.uniform(-0.008, 0.008), MIN_TCP_Z)
    q = (0.0, float(np.cos(yaw / 2)), float(np.sin(yaw / 2)), 0.0)     # top_down_quat(yaw)
    grasp = Pose((obj.pos[0] + off[0], obj.pos[1] + off[1], z), q)
    set_gripper(cell, True)
    if not move_tcp(cell, grasp.offset(dz=0.08)).converged:
        return False
    return move_tcp(cell, grasp, duration=1.0).converged and \
        np.linalg.norm(np.subtract(cell.get_state()["tcp"].pos, grasp.pos)) < 0.005


def collect_demos(seeds: list[int], workers: int = 6,
                  mode: str = "privileged") -> tuple[np.ndarray, np.ndarray, float]:
    """(X, Y, demonstrator success rate) from demonstrator grasps on rehearsal seeds, half on
    the nominal cube and half on the object variants, with DART-style action noise. X holds
    the observation in `mode`; the demonstrator always acts on the privileged view."""
    from gap.rehearsal.runner import map_jobs
    out = map_jobs(_demo_job, [(s, mode) for s in seeds], workers)
    pairs = [p for rec, _ in out for p in rec]
    X, Y = np.array([o for o, _ in pairs]), np.array([a for _, a in pairs])
    return X, Y, float(np.mean([ok for rec, ok in out if rec]))


def _demo_job(job):
    from gap.rehearsal.runner import worker_cell
    from gap.sim.task_sampler import sample
    from gap.skills.grasp import CloseGripper
    seed, mode = job
    rng = np.random.default_rng([seed, 11])
    task = ("place", "stack")[seed % 2]
    inst = sample(task, seed, "object" if (seed // 2) % 2 else "nominal")
    cell = worker_cell(camera_noise=mode == "wrist")   # demos see the deployed wrist camera
    cell.reset(inst)
    cube = str(rng.choice([c for c in mb.CUBES if f"{c}_x" in inst.features]))
    if not demo_start(cell, cube, rng):
        return [], False
    record: list = []
    params = CloseGripper().resolve_params({})
    servo_and_close(cell, params, None, rng, record=record, act_noise=0.3, mode=mode)
    return record, CloseGripper().post(cell, {}, {}, params)[0]
