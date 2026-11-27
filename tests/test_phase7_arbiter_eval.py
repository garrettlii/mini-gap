"""Phase 6b/7 contracts: executor semantics, the arbiter, splits, and run_eval.  --phase 7"""

import json

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


def test_fallback_learned_runs_the_skill_after_the_learned_executor_fails(cell):
    from gap.graph.interpreter import run_graph
    from gap.sim.task_sampler import sample
    from gap.types import SkillResult
    res = run_graph(reference_graph("place"), cell, sample("place", 50_001),
                    executor_policy=lambda n, c, i: "fallback_learned" if n == "close" else "skill",
                    learned_runner=lambda node, c, ins, ps: SkillResult(ok=False, message="no-op"))
    close = next(t for t in res.trace if t.node_id == "close")
    assert close.executor == "fallback_learned" and close.post_ok and res.success


def test_retry_learned_runs_the_learned_executor_twice(cell):
    from gap.graph.interpreter import run_graph
    from gap.sim.task_sampler import sample
    from gap.skills.grasp import CloseGripper
    from gap.types import SkillResult
    calls = []

    def learned(node, c, inputs, params):           # fails once, then closes like the skill
        calls.append(1)
        return CloseGripper().run(c, inputs, params) if len(calls) > 1 else SkillResult(False)
    res = run_graph(reference_graph("place"), cell, sample("place", 50_001),
                    executor_policy=lambda n, c, i: "retry_learned" if n == "close" else "skill",
                    learned_runner=learned)
    close = next(t for t in res.trace if t.node_id == "close")
    assert close.executor == "retry_learned" and len(calls) == 2 and res.success


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


def test_publish_merges_runs(tmp_path):
    import json

    from gap.eval.report import load_cells, publish
    for k in (0, 1):
        (tmp_path / f"run{k}").mkdir()
        cell = {"method": "m", "split": f"s{k}", "task": "place", "successes": k, "n": 2}
        (tmp_path / f"run{k}" / "metrics.json").write_text(json.dumps({"cells": [cell]}))
    out = publish("lab", [str(tmp_path / "run0"), str(tmp_path / "run1")], tmp_path / "res")
    assert [c["split"] for c in load_cells([str(out)])] == ["s0", "s1"]
    redo = {"method": "m", "split": "s0", "task": "place", "successes": 2, "n": 2}
    (tmp_path / "run2").mkdir()
    (tmp_path / "run2" / "metrics.json").write_text(json.dumps({"cells": [redo]}))
    out = publish("lab", [str(tmp_path / f"run{k}") for k in range(3)], tmp_path / "res")
    assert [(c["split"], c["successes"]) for c in load_cells([str(out)])] == [("s0", 2), ("s1", 1)]


def test_loop_metrics_become_before_after_cells(tmp_path):
    import json

    from gap.eval.report import load_cells
    m = {"task": "place", "start": "bug", "bug": "no_release", "heldout_n": 64,
         "heldout_before": 0, "heldout_after": 60}
    (tmp_path / "metrics.json").write_text(json.dumps(m))
    cells = load_cells([str(tmp_path / "metrics.json")])
    assert [(c["method"], c["split"], c["successes"]) for c in cells] == \
        [("start_graph", "no_release", 0), ("after_loop", "no_release", 60)]


def test_wrist_camera_calibration_error_only_with_noise():
    from gap.sim.task_sampler import sample
    from gap.sim.workcell import Workcell
    for noise in (False, True):
        c = Workcell(camera_noise=noise)
        c.reset(sample("place", 50_001))
        pos, _ = c.calibrated_camera("wrist")
        err = np.linalg.norm(pos - c.data.cam_xpos[c.model.camera("wrist").id])
        assert (err > 1e-4) == noise


def test_wrist_observation_finds_the_cube(cell):
    """Accurate while the top face is in view (hand raised); nothing found over bare table.
    At grasp height the top face can leave the image: that is the executor's own failure."""
    from gap.arbiter.fallback import demo_start, observe
    from gap.arbiter.wrist import observe_wrist
    from gap.sim.motion import move_tcp
    from gap.sim.task_sampler import sample
    from gap.types import Pose
    cell.reset(sample("place", 50_002))
    assert demo_start(cell, "red_cube", np.random.default_rng(0))
    tcp = cell.get_state()["tcp"]
    move_tcp(cell, tcp.offset(dz=0.06))
    true, seen = observe(cell, np.random.default_rng(0)), observe_wrist(cell, None)
    assert abs(true[0] - seen[0]) < 0.004 and abs(true[2] - seen[2]) < 0.004
    assert seen[8] > 0
    move_tcp(cell, Pose((0.3, -0.45, 0.3), tcp.quat))     # over bare table
    assert observe_wrist(cell, None)[8] == 0.0


def test_paired_outcomes_become_cells(tmp_path):
    import json

    from gap.eval.report import load_cells
    (tmp_path / "metrics.json").write_text(json.dumps({"paired_rows": 3}))
    (tmp_path / "config.json").write_text(json.dumps({"demos": 25, "obs": "wrist"}))
    rows = [{"obs": [0], "ok_skill": s, "ok_learned": lrn} for s, lrn in
            ((True, False), (True, True), (False, True))]
    (tmp_path / "paired.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    cells = {c["method"]: c["successes"] for c in load_cells([str(tmp_path / "metrics.json")])}
    assert cells == {"skill_only": 1, "learned_only": 1, "both": 1, "neither": 0}


def test_sweep_block_pools_success_and_grasp_time(tmp_path):
    import json

    from gap.eval.report import render
    cells = [{"method": "a@0.1", "split": s, "task": "place", "successes": 5, "n": 10,
              "grasp_s": 3.0, "grasps": 10, "skill_grasps": 4} for s in ("nominal", "pose")]
    (tmp_path / "w.json").write_text(json.dumps({"cells": cells}))
    out = render("<!-- sweep: w -->\n<!-- /sweep -->", tmp_path)
    assert "| a@0.1 | 0.50" in out and "(8/20) | 0.30 |" in out and render(out, tmp_path) == out


def test_executors_read_a_margin_from_the_name(monkeypatch):
    from gap.arbiter import policies
    seen = {}
    monkeypatch.setattr(policies, "load_models", lambda d: (object(), None))
    monkeypatch.setattr(policies, "make_policy",
                        lambda name, g, arb, margin=0.0, **kw: seen.update(name=name, m=margin))
    policies.executors("learned_arbiter@0.2", "x", None, 0)
    assert seen == {"name": "learned_arbiter", "m": 0.2}


@pytest.mark.parametrize("split", ["nominal", "object"])
def test_invert_starts_as_the_stack_goal(cell, split):
    """Green rests on red at reset: stack's success holds, invert's (red on green) does not."""
    from gap.sim import model_builder as mb
    from gap.sim.success import check
    from gap.sim.task_sampler import sample, task_config
    inst = sample("invert", 50_000, split)
    half = {c: mb.half_height(int(inst.features.get(f"{c}_variant", 0))) for c in mb.CUBES}
    state = cell.reset(inst)                     # raises if the stack does not settle
    tasks = task_config()["tasks"]
    assert check("stack", state, tasks["stack"], half)
    assert not check("invert", state, tasks["invert"], half)
    g, r = inst.object_poses["green_cube"].pos, inst.object_poses["red_cube"].pos
    assert g[:2] == r[:2] and g[2] == pytest.approx(2 * half["red_cube"] + half["green_cube"])


def test_unstack_starts_as_a_three_cube_tower(cell):
    """Blue on green on red at reset; no cube is in its assigned zone yet."""
    from gap.sim import model_builder as mb
    from gap.sim.success import check
    from gap.sim.task_sampler import sample, task_config
    inst = sample("unstack", 50_000, "object")
    half = {c: mb.half_height(int(inst.features.get(f"{c}_variant", 0))) for c in mb.CUBES}
    state = cell.reset(inst)                     # raises if the tower does not settle
    assert not check("unstack", state, task_config()["tasks"]["unstack"], half)
    r, g, b = (inst.object_poses[f"{c}_cube"].pos for c in ("red", "green", "blue"))
    assert r[:2] == g[:2] == b[:2]
    assert b[2] == pytest.approx(2 * half["red_cube"] + 2 * half["green_cube"] + half["blue_cube"])


def test_reset_lets_a_leaning_tower_settle(cell):
    """Dev seed 50010, object split: a cylinder on a 3 cm cube leans and carries the top cube
    past 1 mm; the tolerance is 1 mm per contact beneath, so reset accepts it."""
    from gap.sim.task_sampler import sample
    inst = sample("unstack", 50_010, "object")
    cell.reset(inst)
    top = np.linalg.norm(cell.data.body("blue_cube").xpos - np.array(inst.object_poses["blue_cube"].pos))
    assert 1e-3 < top < 3e-3


def test_paired_comparison_counts_discordant_episodes(tmp_path):
    import json

    from gap.eval.report import add_comparison, paired, render
    rows = [{"method": m, "split": "nominal", "task": "place", "seed": s, "success": ok}
            for s in range(10) for m, ok in (("a", s < 9), ("b", s < 1))]
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    comp = paired([str(tmp_path / "run")], "a", "b")
    assert (comp["episodes"], comp["a_only"], comp["b_only"]) == (10, 8, 0)
    assert comp["p"] == pytest.approx(2 * 0.5**8)
    add_comparison("cmp", comp, tmp_path)
    out = render("<!-- paired: cmp -->\n<!-- /paired -->", tmp_path)
    assert "| a vs b | 10 | 8 | 0 |" in out and render(out, tmp_path) == out
    (tmp_path / "later").mkdir()                 # a later run's "a" replaces the first run's
    (tmp_path / "later" / "episodes.jsonl").write_text("".join(
        json.dumps({**r, "success": True}) + "\n" for r in rows if r["method"] == "a"))
    later = paired([str(tmp_path / "run"), str(tmp_path / "later")], "a", "b")
    assert (later["a_only"], later["b_only"]) == (9, 0)


def test_retry_graph_adds_a_fail_edge_from_every_grasp_node():
    from gap.arbiter.policies import arbiter_graph, retry_graph
    from gap.graph.validate import validate
    for task in ("place", "unstack"):
        g, base = retry_graph(task), arbiter_graph(task)
        closes = [n.id for n in g.nodes if n.skill == "close_gripper"]
        new = [e for e in g.edges if e not in base.edges]
        assert sorted(e.src for e in new) == sorted(closes)
        assert all(e.on == "fail" and e.to.startswith("find") for e in new)
        assert not validate(g)


def test_seq_fallback_runs_first_the_order_the_arbiter_picks(cell):
    from gap.arbiter.policies import arbiter_graph, make_policy
    from gap.sim.task_sampler import sample

    class Pick:
        def __init__(self, first):
            self.first = first

        def choose(self, x, margin=0.0):
            return self.first
    g = arbiter_graph("place")
    cell.reset(sample("place", 50_000))
    close = next(n.id for n in g.nodes if n.skill == "close_gripper")
    for first, executor in (("skill", "fallback"), ("learned", "fallback_learned")):
        assert make_policy("seq_fallback", g, Pick(first))(close, cell, None) == executor


def test_render_table_keeps_only_the_named_splits(tmp_path):
    from gap.eval.report import render
    cells = [{"method": "m", "split": s, "task": "place", "successes": k, "n": 10}
             for s, k in (("nominal", 5), ("distractors", 10))]
    (tmp_path / "lab.json").write_text(json.dumps({"cells": cells}))
    out = render("<!-- table: lab by=task splits=nominal --><!-- /table -->", tmp_path)
    assert "(5/10)" in out and "15/20" not in out
