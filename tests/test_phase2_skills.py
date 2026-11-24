"""Phase 2 contracts: skills.  uv run pytest --phase 2

Add one test per skill as you implement it: 20 dev seeds, assert its post-check.
"""

import numpy as np
import pytest

import gap.skills as skills
from gap.skills.base import Skill

pytestmark = pytest.mark.phase(2)


@pytest.mark.parametrize("name", skills.names())
def test_every_skill_has_a_post_check(name):
    cls = skills.get(name)
    assert cls.post is not Skill.post, f"{name} must override post()"


def test_locate_object_gt_matches_state():
    from gap.sim.task_sampler import sample
    from gap.sim.workcell import Workcell
    cell, skill = Workcell(), skills.get("locate_object_gt")()
    for seed in range(50000, 50020):
        s = cell.reset(sample("place", seed=seed))
        res = skill.run(cell, {"color": "red"}, skill.resolve_params())
        assert res.ok
        assert np.linalg.norm(res.outputs["pose"].p - s["objects"]["red_cube"].p) < 1e-3
        assert skill.post(cell, {"color": "red"}, res.outputs, {})[0]


PLACE_CHAIN = [  # the Place oracle as skill calls; refs are "<node>.<port>"
    ("find", "locate_object_gt", {"color": "red"}, {}),
    ("plan", "plan_top_grasp", {"pose": "find.pose"}, {}),
    ("above", "move_to_pose", {"pose": "plan.pre_pose"}, {}),
    ("open", "open_gripper", {}, {}),
    ("reach", "move_to_pose", {"pose": "plan.grasp_pose"}, {"speed": 0.3}),
    ("close", "close_gripper", {}, {}),
    ("check", "verify_grasp", {"color": "red"}, {}),
    ("target", "compute_place_pose", {"target": "zone_a"}, {}),
    ("carry", "move_above", {"pose": "target.pose"}, {"height": 0.10}),
    ("lower", "move_to_pose", {"pose": "target.pose"}, {"speed": 0.3}),
    ("release", "open_gripper", {}, {}),
    ("leave", "retreat", {}, {}),
]


def run_chain(cell, chain):
    """Run skills in order; return the first failing node id, or None."""
    out = {}
    for nid, name, ins, ps in chain:
        skill = skills.get(name)()
        ins = {k: out[v.split(".")[0]][v.split(".")[1]] if isinstance(v, str) and "." in v else v
               for k, v in ins.items()}
        params = skill.resolve_params(ps)
        res = skill.run(cell, ins, params)
        if not (res.ok and skill.post(cell, ins, res.outputs, params)[0]):
            return nid
        out[nid] = res.outputs
    return None


def test_place_chain_every_post_check_passes():
    from gap.sim.success import settled_success
    from gap.sim.task_sampler import sample
    from gap.sim.workcell import Workcell
    cell = Workcell()
    for seed in range(50000, 50010):
        cell.reset(sample("place", seed=seed))
        assert run_chain(cell, PLACE_CHAIN) is None, f"seed {seed}"
        assert settled_success(cell), f"seed {seed}"


def test_locate_object_rgbd_within_1cm():
    from gap.sim.task_sampler import sample
    from gap.sim.workcell import Workcell
    cell, skill = Workcell(), skills.get("locate_object_rgbd")()
    for seed in range(50000, 50020):
        s = cell.reset(sample("place", seed=seed))
        res = skill.run(cell, {"color": "red"}, skill.resolve_params())
        assert res.ok and skill.post(cell, {"color": "red"}, res.outputs, {})[0]
        assert np.linalg.norm(res.outputs["pose"].p - s["objects"]["red_cube"].p) < 0.01
        assert np.allclose(cell.data.ctrl[:7], s["qpos_arm"], atol=1e-6), "arm returns after the picture"


def test_post_checks_catch_failures():
    from gap.sim.task_sampler import sample
    from gap.sim.workcell import Workcell
    cell = Workcell()
    cell.reset(sample("place", seed=50000))
    close = skills.get("close_gripper")()
    close.run(cell, {}, close.resolve_params())
    ok, why = close.post(cell, {}, {}, {})
    assert not ok and "nothing" in why, "closing on air must fail the post-check"
    verify = skills.get("verify_grasp")()
    params = verify.resolve_params()
    verify.run(cell, {"color": "red"}, params)
    assert not verify.post(cell, {"color": "red"}, {}, params)[0], "no cube rose with the hand"
    find = skills.get("locate_object_gt")()
    assert not find.run(cell, {"color": "purple"}, {}).ok


def test_compute_place_pose_on_a_cube():
    from gap.sim import model_builder as mb
    from gap.sim.task_sampler import sample
    from gap.sim.workcell import Workcell
    cell = Workcell()
    s = cell.reset(sample("stack", seed=50000))
    skill = skills.get("compute_place_pose")()
    params = skill.resolve_params()
    pose = skill.run(cell, {"target": "red"}, params).outputs["pose"]
    red = s["objects"]["red_cube"].p
    assert np.allclose(pose.p[:2], red[:2], atol=1e-6)
    assert pose.p[2] == pytest.approx(red[2] + 2 * mb.CUBE_HALF + params["drop_height"], abs=1e-6)
