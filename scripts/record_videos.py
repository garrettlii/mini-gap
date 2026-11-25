"""Record the three demo clips on dev seeds and join them into one captioned video.

  uv run python scripts/record_videos.py --models runs/<train_arbiter> --repaired runs/<loop run>
  uv run python scripts/record_videos.py --models runs/<train_arbiter> --repaired runs/<gen_repairs>#2

1. success: the stack graph with the learned arbiter, camera noise on
2. recovered failure: a seeded-bug graph fails; the loop's repaired graph succeeds on the same
   seed (a run_loop dir's final_graph.json, or problem #i of a gen_repairs run; both were
   rehearsed with the clean sensor)
3. arbiter switching: sort on the object split; the caption names the executor at each node
Each clip uses the first dev seed (from 50,000) whose episode shows what the clip is about.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gap.arbiter.policies import arbiter_graph, executors
from gap.graph.edits import SetParam, apply_edit
from gap.graph.interpreter import run_graph
from gap.graph.schema import Graph
from gap.rehearsal.seeded_bugs import buggy_graph, reference_graph
from gap.sim.task_sampler import sample
from gap.sim.workcell import VIDEO_FPS, Workcell
from gap.utils.runs import RunDir, read_jsonl
from gap.utils.seeds import seed_range
from gap.utils.video import save_mp4

FONT = ImageFont.load_default(size=20)
HOLD = VIDEO_FPS          # one second on the last frame of each episode
MAX_TRIES = 40


def episode(cell, graph, inst, policy=None, models=None, frames=None):
    """Run one episode; marks = [(frame index, node id, executor)] at every node start."""
    marks: list = []
    pol, learned = executors(policy, models, graph, inst.seed) if policy else (None, None)

    def logged(node_id, c, i):
        choice = pol(node_id, c, i) if pol else "skill"
        marks.append((len(frames) if frames is not None else 0, node_id, choice))
        return choice
    res = run_graph(graph, cell, inst, executor_policy=logged, learned_runner=learned,
                    frame_sink=frames)
    return res, marks


def find_seed(cell, graph_fn, task, split, want, **kw) -> int:
    """First dev seed whose (unrecorded) episode satisfies want(result, marks)."""
    for seed in range(seed_range("dev").start, seed_range("dev").start + MAX_TRIES):
        inst = sample(task, seed, split)
        if want(*episode(cell, graph_fn(), inst, **kw)):
            return seed
    raise SystemExit(f"no dev seed in {MAX_TRIES} tries for {task}/{split}")


def captioned(cell, graph, inst, title, **kw) -> tuple[list, dict]:
    frames: list = []
    res, marks = episode(cell, graph, inst, frames=frames, **kw)
    out = []
    for k, f in enumerate(frames + [frames[-1]] * HOLD):
        cur = [m for m in marks if m[0] <= k]
        sub = f"node {cur[-1][1]}: {cur[-1][2]}" if cur and k < len(frames) else \
            ("result: success" if res.success else "result: failed")
        out.append(_draw(f, title, sub))
    info = {"seed": inst.seed, "task": inst.task, "split": inst.split, "success": res.success,
            "executors": [(n, e) for _, n, e in marks]}
    return out, info


def repair_case(spec: str) -> tuple[str, Graph, Graph, str]:
    """(task, buggy graph, repaired graph, bug label) from "<run_loop dir>" or
    "<gen_repairs dir>#<problem index>" (problems.jsonl and loop_summary.jsonl align)."""
    if "#" not in spec:
        cfg = json.loads(Path(spec, "config.json").read_text())
        final = Graph.model_validate_json(Path(spec, "final_graph.json").read_text())
        return cfg["task"], buggy_graph(cfg["task"], cfg["bug"]), final, cfg["bug"]
    run, idx = spec.split("#")
    prob = read_jsonl(Path(run, "problems.jsonl"))[int(idx)]
    final = Graph.model_validate_json(read_jsonl(Path(run, "loop_summary.jsonl"))[int(idx)]["final_graph"])
    bug = SetParam(**prob["bug"])
    return (prob["task"], apply_edit(reference_graph(prob["task"]), bug), final,
            f"{bug.node}.{bug.param}={bug.value}")


def _draw(frame: np.ndarray, title: str, sub: str) -> np.ndarray:
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    w, h = img.size
    d.rectangle([0, 0, w, 32], fill=(0, 0, 0))
    d.text((10, 6), title, fill=(255, 255, 255), font=FONT)
    d.rectangle([0, h - 32, w, h], fill=(0, 0, 0))
    d.text((10, h - 26), sub, fill=(255, 255, 255), font=FONT)
    return np.asarray(img)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="train_arbiter run dir")
    ap.add_argument("--repaired", required=True, help="run_loop run dir (final_graph.json)")
    args = ap.parse_args()
    task, buggy, repaired, bug = repair_case(args.repaired)
    run = RunDir("videos", vars(args))
    noisy, clean = Workcell(camera_noise=True, video_size=(480, 640)), Workcell(video_size=(480, 640))
    arb = {"policy": "learned_arbiter", "models": args.models}
    clips, infos = [], []

    seed = find_seed(noisy, lambda: arbiter_graph("stack"), "stack", "nominal",
                     lambda r, m: r.success, **arb)
    c, i = captioned(noisy, arbiter_graph("stack"), sample("stack", seed),
                     "1/3 stack, learned arbiter, noisy camera", **arb)
    clips.append(c)
    infos.append(i)

    def fixed_only(r, m):
        return r.success and not episode(clean, buggy, sample(task, r.instance.seed))[0].success
    seed = find_seed(clean, lambda: repaired, task, "nominal", fixed_only)
    for graph, title in ((buggy, f"2/3 {task}, seeded bug: {bug}"),
                         (repaired, f"2/3 {task}, after the repair loop")):
        c, i = captioned(clean, graph, sample(task, seed), title)
        clips.append(c)
        infos.append(i)

    def switches(r, m):
        grasp = {e for _, n, e in m if n.startswith("close")}
        return r.success and "learned" in grasp
    seed = find_seed(noisy, lambda: arbiter_graph("sort"), "sort", "object", switches, **arb)
    c, i = captioned(noisy, arbiter_graph("sort"), sample("sort", seed, "object"),
                     "3/3 sort, object variants, arbiter picks per node", **arb)
    clips.append(c)
    infos.append(i)

    for k, (frames, info) in enumerate(zip(clips, infos, strict=True)):
        info["file"] = str(save_mp4(frames, run.file(f"clip{k}.mp4"), VIDEO_FPS).name)
        run.append_jsonl("clips.jsonl", info)
    print(save_mp4([f for c in clips for f in c], run.file("demo.mp4"), VIDEO_FPS))
    run.metrics(clips=infos, seconds=sum(len(c) for c in clips) / VIDEO_FPS)


if __name__ == "__main__":
    main()
