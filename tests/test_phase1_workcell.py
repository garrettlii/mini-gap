"""Phase 1 contracts: workcell, sampler, IK, success checks, oracle.  uv run pytest --phase 1"""

import numpy as np
import pytest

from gap.sim import model_builder as mb
from gap.types import Pose

pytestmark = pytest.mark.phase(1)


@pytest.fixture
def cell():
    from gap.sim.workcell import Workcell
    c = Workcell()
    yield c
    c.close()


def test_state_has_contract_keys(cell):
    from gap.sim.task_sampler import sample
    s = cell.reset(sample("place", seed=50000))
    assert set(s) >= {"time", "tcp", "gripper_width", "objects", "contacts", "qpos_arm"}
    assert isinstance(s["tcp"], Pose) and len(s["qpos_arm"]) == 7
    assert 0.07 < s["gripper_width"] < 0.085, "home pose has the gripper open"


def test_reset_places_objects(cell):
    from gap.sim.task_sampler import sample
    inst = sample("place", seed=50001)
    s = cell.reset(inst)
    for name, pose in inst.object_poses.items():
        assert np.linalg.norm(s["objects"][name].p - pose.p) < 2e-3


def test_reset_rejects_overlapping_cubes(cell):
    from gap.sim.task_sampler import sample
    inst = sample("place", seed=50005)
    inst.object_poses["green_cube"] = inst.object_poses["red_cube"].offset(dx=0.01)  # 3 cm overlap
    with pytest.raises(RuntimeError, match="spawned overlapping"):
        cell.reset(inst)


def test_step_rejects_bad_shape_and_clips_ctrl(cell):
    from gap.sim.task_sampler import sample
    cell.reset(sample("place", seed=50006))
    with pytest.raises(ValueError):
        cell.step(0.5)  # a scalar must not broadcast to all 8 actuators
    ctrl = cell.data.ctrl.copy()
    ctrl[7] = 999.0
    cell.step(ctrl)
    assert cell.data.ctrl[7] == mb.GRIPPER_OPEN_CTRL, "stored ctrl must equal the applied (clamped) value"


def test_sampler_is_deterministic_and_in_envelope():
    from gap.sim.task_sampler import sample, task_config
    env = task_config()["envelope"]
    a, b = sample("stack", seed=50002), sample("stack", seed=50002)
    assert a.object_poses == b.object_poses
    for seed in range(50000, 50050):
        inst = sample("stack", seed=seed)
        used = task_config()["tasks"]["stack"]["objects"]
        ps = [inst.object_poses[o].p for o in used]
        for p in ps:
            assert env["x"][0] <= p[0] <= env["x"][1] and env["y"][0] <= p[1] <= env["y"][1]
        assert np.linalg.norm(ps[0][:2] - ps[1][:2]) >= env["min_separation"]
        assert inst.features, "fill features for diagnosis and the arbiter"


def test_snapshot_restore_is_exact(cell):
    from gap.sim.task_sampler import sample
    cell.reset(sample("place", seed=50003))
    ctrl = cell.data.ctrl.copy()
    ctrl[0] += 0.3
    snap = cell.snapshot()
    a = cell.step(ctrl, 200)
    cell.restore(snap)
    b = cell.step(ctrl, 200)
    assert a["qpos_arm"] == b["qpos_arm"] and a["time"] == b["time"]


def test_ik_reaches_random_targets(cell):
    from gap.sim.ik import solve_ik, top_down_quat
    from gap.sim.task_sampler import sample
    cell.reset(sample("place", seed=50004))
    rng = np.random.default_rng(0)
    for _ in range(20):
        target = Pose((rng.uniform(0.35, 0.65), rng.uniform(-0.25, 0.25), rng.uniform(0.03, 0.3)),
                      top_down_quat(rng.uniform(-np.pi, np.pi)))
        r = solve_ik(cell.model, cell.data, target)
        assert r.converged and r.pos_err < 5e-3, f"IK failed for {target}"


def test_success_checks_on_constructed_states():
    from gap.sim.success import check
    from gap.sim.task_sampler import task_config
    cfg = task_config()["tasks"]
    zone = task_config()["zones"]["zone_a"]["pos"]
    base = {"time": 0.0, "tcp": Pose((0.4, 0, 0.3)), "gripper_width": 0.08, "contacts": [],
            "qpos_arm": [0.0] * 7}
    in_zone = {**base, "objects": {"red_cube": Pose((zone[0], zone[1], mb.CUBE_HALF))}}
    off_zone = {**base, "objects": {"red_cube": Pose((0.5, 0.0, mb.CUBE_HALF))}}
    assert check("place", in_zone, cfg["place"])
    assert not check("place", off_zone, cfg["place"])


@pytest.mark.slow
def test_place_oracle_95_percent():
    from gap.sim.oracle import run_place_oracle
    from gap.sim.task_sampler import sample
    from gap.sim.workcell import Workcell
    from gap.utils.stats import fmt_rate
    cell = Workcell()
    wins = sum(run_place_oracle(cell, sample("place", seed=s)) for s in range(50000, 50100))
    print("oracle place:", fmt_rate(wins, 100))
    assert wins >= 95
