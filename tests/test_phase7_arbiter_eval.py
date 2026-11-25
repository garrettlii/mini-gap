"""Phase 6b/7 contracts: executor semantics, the arbiter, splits, and run_eval.  --phase 7"""

import numpy as np
import pytest

from gap.rehearsal.seeded_bugs import reference_graph

pytestmark = pytest.mark.phase(7)


@pytest.fixture(scope="module")
def cell():
    from gap.sim.workcell import Workcell
    return Workcell()


def _run(cell, executor, monkeypatch, calls):
    """Place episode where the scripted close always fails its pre-check."""
    from gap.graph.interpreter import run_graph
    from gap.sim.task_sampler import sample
    from gap.skills.grasp import CloseGripper
    monkeypatch.setattr(CloseGripper, "pre", lambda self, c, i, p: (False, "forced"))

    def learned(node, c, inputs, params):
        calls.append(c.get_state()["gripper_width"])
        return CloseGripper().run(c, inputs, params)

    def policy(node_id, c, inst):
        return executor if node_id == "close" else "skill"
    return run_graph(reference_graph("place"), cell, sample("place", 50_001),
                     executor_policy=policy, learned_runner=learned)


@pytest.mark.parametrize("executor", ["skill", "learned", "fallback", "oracle"])
def test_executor_values(cell, monkeypatch, executor):
    calls: list = []
    res = _run(cell, executor, monkeypatch, calls)
    close = next(t for t in res.trace if t.node_id == "close")
    assert close.executor == executor
    assert res.success == (executor != "skill"), "only the learned path survives a failed pre"
    assert len(calls) == (executor != "skill")


def test_arbiter_picks_the_executor_that_wins_in_that_region():
    from gap.arbiter.arbiter import Arbiter
    rng = np.random.default_rng(0)
    X = np.zeros((400, 8))
    X[:, 0] = rng.uniform(-0.02, 0.02, 400)            # close-axis offset
    X[:, 4] = 0.02
    y_skill = np.abs(X[:, 0]) < 0.008                  # skill only works near center
    y_learned = rng.random(400) < 0.9
    arb = Arbiter().fit(X, y_skill, y_learned)
    centered, off = X[np.argmin(np.abs(X[:, 0]))], X[np.argmax(np.abs(X[:, 0]))]
    assert arb.choose(centered) == "skill"
    assert arb.choose(off) == "learned"


def test_object_split_variants_rest_on_the_table():
    from gap.sim import model_builder as mb
    from gap.sim.task_sampler import sample
    seen = set()
    for seed in range(50_000, 50_040):
        inst = sample("stack", seed, "object")
        for cube in ("red_cube", "green_cube"):
            v = int(inst.features[f"{cube}_variant"])
            seen.add(mb.VARIANT_NAMES[v])
            assert inst.object_poses[cube].pos[2] == pytest.approx(mb.half_height(v))
    assert seen == {"cube_3cm", "cube_5cm", "cylinder"}


def test_camera_noise_shifts_rgbd_estimates_by_millimeters(cell):
    from gap.sim.task_sampler import sample
    from gap.skills.perception import LocateObjectRGBD
    errs = []
    for seed in range(50_000, 50_010):
        inst = sample("place", seed)
        cell.camera_noise = True
        cell.reset(inst)
        sk = LocateObjectRGBD()
        pose = sk.run(cell, {"color": "red"}, sk.resolve_params({})).outputs["pose"]
        cell.camera_noise = False
        true = np.asarray(inst.object_poses["red_cube"].pos[:2])
        errs.append(np.linalg.norm(np.asarray(pose.pos[:2]) - true))
    assert 0.002 < np.mean(errs) < 0.03


@pytest.mark.slow
def test_run_eval_writes_cells(tmp_path, monkeypatch):
    import json
    import pickle

    from gap.arbiter.fallback import FallbackPolicy
    from gap.eval import run_eval
    from gap.utils import runs
    rng = np.random.default_rng(0)
    bc = FallbackPolicy("grasp").fit(rng.normal(size=(64, 8)), rng.normal(size=(64, 5)))
    (tmp_path / "bc_grasp.pkl").write_bytes(pickle.dumps(bc))
    monkeypatch.setattr(runs.RunDir.__init__, "__defaults__", (tmp_path / "runs",))
    run_eval.main(["--methods", "always_skill", "--splits", "nominal", "--n", "2",
                   "--models", str(tmp_path)])
    metrics = json.loads(next((tmp_path / "runs").glob("*/metrics.json")).read_text())
    assert [(c["task"], c["n"]) for c in metrics["cells"]] == [("place", 2), ("stack", 2)]


def test_render_refills_marked_blocks_idempotently(tmp_path):
    import json

    from gap.eval.report import render
    cells = [{"method": "m", "split": "nominal", "task": t, "successes": 8, "n": 10}
             for t in ("place", "stack")]
    (tmp_path / "r.json").write_text(json.dumps({"cells": cells}))
    text = "x <!-- rate: r m nominal -->old<!-- /rate --> y\n<!-- table: r by=task -->\n<!-- /table -->"
    once = render(text, tmp_path)
    assert render(once, tmp_path) == once
    assert "(16/20)" in once and "| m | 0.80" in once and once.startswith("x <!-- rate")


def test_eval_swaps_out_the_ground_truth_locator(monkeypatch):
    from gap.eval import run_eval
    seen = {}
    monkeypatch.setattr(run_eval, "rehearse", lambda g, *a, **k: seen.setdefault("graph", g))
    run_eval._rehearse(reference_graph("place"), [], True)
    skills = {n.skill for n in seen["graph"].nodes}
    assert "locate_object_rgbd" in skills and "locate_object_gt" not in skills
