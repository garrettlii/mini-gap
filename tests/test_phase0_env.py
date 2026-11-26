"""Phase 0: environment and contracts that are already implemented. Must pass after setup."""

import platform

import mujoco
import numpy as np
import pytest

from gap.sim import model_builder as mb


def test_workcell_compiles(model):
    assert model.nu == 8, "7 arm actuators + 1 gripper"
    for name in mb.CUBES:
        model.body(name)
    for name in mb.ZONES:
        model.site(name)
    for cam in mb.CAMERAS:
        model.camera(cam)
    model.site(mb.TCP_SITE)
    model.key(mb.HOME_KEY)


def test_home_pose_is_stable(model):
    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d, model.key(mb.HOME_KEY).id)
    mujoco.mj_forward(model, d)
    start = {c: d.body(c).xpos.copy() for c in mb.CUBES}
    for _ in range(1000):  # 2 s of sim time
        mujoco.mj_step(model, d)
    for c in mb.CUBES:
        assert np.linalg.norm(d.body(c).xpos - start[c]) < 1e-3, f"{c} drifted"
        assert abs(d.body(c).xpos[2] - mb.CUBE_HALF) < 2e-3, f"{c} not resting on table"
    q_home = model.key(mb.HOME_KEY).qpos[:7]
    assert np.abs(d.qpos[:7] - q_home).max() < 0.02, "arm sagged under its position controller"


def test_offscreen_render(model):
    try:
        r = mujoco.Renderer(model, 120, 160)
    except Exception as e:  # noqa: BLE001 -- headless CI without GL raises varied errors
        pytest.skip(f"no offscreen GL context: {e}")
    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d, model.key(mb.HOME_KEY).id)
    mujoco.mj_forward(model, d)
    for cam in mb.CAMERAS:
        r.update_scene(d, camera=cam)
        img = r.render()
        assert img.shape == (120, 160, 3) and img.std() > 5, f"{cam} rendered blank"
    r.close()


@pytest.mark.skipif(platform.system() != "Darwin" or platform.machine() != "arm64",
                    reason="MLX needs Apple Silicon")
def test_mlx_available():
    import mlx.core as mx
    assert mx.sum(mx.array([1, 2, 3])).item() == 6


def test_seed_sets_are_disjoint():
    from gap.utils import seeds
    seeds.assert_disjoint()
    ev = seeds.eval_seeds("pose", 100)
    assert len(ev) == 100 and all(seeds.is_eval_seed(s) for s in ev)
    assert not any(seeds.is_eval_seed(s) for s in seeds.rehearsal_seeds(100))
    assert seeds.eval_seeds("nominal")[0] != seeds.eval_seeds("pose")[0]


def test_wilson_interval():
    from gap.utils.stats import wilson_interval
    lo, hi = wilson_interval(83, 100)
    assert 0.74 < lo < 0.75 and 0.89 < hi < 0.90
    assert wilson_interval(0, 10)[0] == 0.0


def test_skill_catalog():
    from gap import skills
    assert len(skills.names()) == 10
    text = skills.catalog_text()
    assert "plan_top_grasp" in text and "approach_height" in text
    p = skills.get("plan_top_grasp")().resolve_params({"approach_height": 0.1})
    assert p["approach_height"] == 0.1
    with pytest.raises(ValueError):
        skills.get("plan_top_grasp")().resolve_params({"approach_height": 5.0})


def test_example_graph_parses():
    from pathlib import Path

    from gap.graph.schema import Graph
    g = Graph.model_validate_json(Path("graphs/place_example.json").read_text())
    assert g.parse_ref("find.pose") == ("find", "pose")
    assert g.parse_ref("red") is None
    assert g.next_on_success("find") == "plan"
    assert g.next_on_success(g.nodes[-1].id) is None
    assert g.fail_edge("check").max_retries == 2


def test_edit_ops_parse():
    from pydantic import ValidationError

    from gap.graph.edits import parse_edit
    e = parse_edit('{"op": "set_param", "node": "close", "param": "force", "value": 0.8}')
    assert e.op == "set_param" and e.value == 0.8
    with pytest.raises(ValidationError):
        parse_edit('{"op": "teleport", "node": "close"}')
