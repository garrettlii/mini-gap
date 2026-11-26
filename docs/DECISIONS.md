# Decision log

Record any change to a contract (gap/types.py, skill base, graph semantics, seeds, eval
splits) before making it. Format: date, decision, why, what it affects.

## 2026-11-01: scaffold defaults
- Graph success transitions default to next-in-list; only fail edges must be explicit.
  Why: shorter graphs, fewer tokens for Qwen to learn. Affects prompts, validator, SFT data.
- TCP site and wrist camera are added in Python (MjSpec), not XML.
  Why: they attach to the hand body inside Menagerie's panda.xml, which an including file can't edit.
- Tabletop at z = 0 with the robot base on it; overhead camera is occluded in the home pose,
  so perception needs a camera-clear arm pose first (as GaP stages its cameras).

## 2026-11-24: local LLM only
- No paid API calls. The planner, specialists, and loop run on local Qwen (configs/models.yaml
  `local`); the frontier rows of the Phase 6 and Phase 7 tables are dropped. Phase 6 SFT data
  comes from the local model's own successful edits (rejection sampling) instead of frontier logs.

## 2026-11-24: Phase 1 design choices
- IK: when a joint sits at a limit and the step pushes it further, its Jacobian column is
  zeroed and the step re-solved. If a start fails, retry with joint 7 seeded at
  (-2.5, -1.25, 0, 1.25, 2.5). Why: the wrist turns less than a full circle, so the
  shortest-rotation step can drive it into its limit when the target is the long way round.
- top_down_quat(yaw): yaw is the TCP x-axis heading; fingers close along yaw - pi/2; home = pi/2.
- success.check is static on one State. The at-rest requirement is settled_success(cell):
  hold ctrl 0.2 s and require < 1 mm drift. Interpreter and oracle call settled_success.
  Thresholds: resting = |z - expected| <= 5 mm; gripper open = width >= 4.5 cm.
- gap/sim/motion.py holds the ramped joint-space moves (smoothstep over 20 ms increments) so the
  oracle and the Phase 2 skills share one implementation. Oracle takes frame_sink (a list),
  like run_graph, instead of record=True.

## 2026-11-24: gravity compensation on the arm (model_builder.py)
- Arm links, hand, and fingers get gravcomp = 1. Why: the position servos sagged 6.7 mm at
  home under gravity, so move_to_pose missed its 5 mm tolerance. Franka's real controller
  compensates gravity too. Affects all physics; Phase 1 tests re-run after the change.
- close_gripper.force: ctrl = 127.5 * (1 - force), so force is the fraction of the max squeeze
  on a 4 cm cube. Why: 255 * (1 - force) put the default 0.6 at a 3.2 cm target, a 0.4 N squeeze
  that left one finger off the cube.
- Gripper actuator 10x stiffer (GRIP_GAIN; damping x sqrt(10)). Why: Menagerie's ~1 N per
  finger on a 4 cm cube is below the cube's table friction (0.74 N), so any grasp 0.5 mm off
  center stalled with one finger touching (0/20 at 0.5 mm offset). RGBD place went 14/30 ->
  30/30 on dev seeds; GT unchanged at 30/30.

## 2026-11-24: Phase 2-3 design choices
- move_joints holds at the goal until every joint is within 1 mrad (0.1 s min, 1 s max)
  instead of a fixed 0.2 s. Why: the servos are overdamped and creep in after long moves;
  a fixed hold left ~6 mm TCP error on 4/30 sort episodes.
- compute_place_pose picks the zone slot (center or a corner 4.5 cm out on each axis) farthest
  from cubes already resting in the zone. Why: sort puts two cubes in zone A; a single center
  spot stacked them. Description updated so agents know.
- locate_object_rgbd swings the arm to a camera-clear pose for the picture and back. Why: the
  arm blocks the overhead camera at home, and leaving it aside seeded later IK in a twisted pose.
- Recording lives in Workcell.step (cell.frame_sink) rather than threaded through helpers.
- Interpreter: fail-edge budget counts fail-edge traversals per source node, not visits.
  run_graph takes learned_runner(node, cell, inputs, params) for the Phase 6b learned executor.
  Episode sim_steps come from the sim clock, not SkillResult.sim_steps.
- Example graphs for stack and sort are in graphs/ (dev-seed check: place 50/50, stack 50/50,
  sort 48/50 as of this commit).

## 2026-11-24: Phase 4 methods and prompts
- Methods: `code` (free-form Python over the skills, run by gap/graph/code_exec.py with the
  same pre/post checks and success check), `graph_single` (planner with up to 3 validator-
  feedback attempts), `graph_multi` (planner, then one critic review whose structural edits are
  kept only if the graph still validates). Defined in gap/agents/methods.py.
- Every prompt carries ONE worked example for a task outside the task set ("pick up the blue
  cube and hold it above zone B"), as a graph for planners and as code for the coder. Why: a 2B
  model cannot assemble a 12-node graph from the catalog alone; both methods get the same help.
- Code policies run with a builtin whitelist, no imports, <= 60 skill calls, and <= 100k policy
  lines. Not a security sandbox; acceptable because the code comes from the local model.
- Prompts are filled with context.fill (plain replace), not str.format, so literal JSON braces
  in prompt files survive.

## 2026-11-24: two local models, and the Phase 5 loop
- configs/models.yaml gets `planner_repo` (Qwen3.5-4B-4bit) beside `mlx_repo` (2B). The 4B
  writes whole graphs and code and plays the critic; the 2B proposes single-node edits and
  is the model Phase 6 trains. Why: the 2B could not produce a valid 12-node graph in 3
  attempts (repetition loops, invented ports); the 4B produced valid place and stack graphs
  on the first attempt (~60 s each once loaded). It also stands in for the dropped frontier rows.
- MLXBackend reseeds mx.random before every call, so a repeated (prompt, temperature) gives the
  same sample. Planner, coder, and specialist retry at temperature > 0 so retries differ.
- Planner output is parsed from the first complete JSON object (raw_decode); trailing text
  is ignored. "in"/"params" entries the model fills with null or {} are dropped before
  validation. Validator feedback lists at most a few short errors.
- Loop routing: for the most-blamed node, the specialist is asked twice about that node, then
  once about each upstream suspect (for grasp failures: earlier grasp-pipeline nodes and the
  nodes that consume their outputs; always: data-flow ancestors), at most 4 nodes. The critic
  then gets the full report. Why: a too-deep grasp shows up as a failed close/verify, not at
  plan_top_grasp, so asking only the blamed node can never find the fix.
- Seeded bugs (gap/rehearsal/seeded_bugs.py) give a repair benchmark with a known answer:
  diagonal_grasp, grasp_too_deep, tight_verify, no_release. Each breaks place, stack, and
  sort (0/12 dev seeds each). scripts/run_loop.py optimizes on one block of rehearsal seeds
  and reports a disjoint block (offset 40,000) as held-out.

## 2026-11-24: Phase 6 data and training
- Failure reports for grasp failures add a measured line: the hand's yaw against the nearest
  cube's faces and the fingertip offset from the cube center at the failing node. Why: the
  error alone ("finger width 0.052 m outside [0.03, 0.045]") gave no route to yaw_offset.
- The specialist prompt lists each param's doc string for its node's skill (catalog_entry
  itself is unchanged, so planner prompts stay the same length).
- The loop skips an edit already tried on that node instead of rehearsing it again.
- The critic prompt shows the exact JSON form of each structural op (it was inventing keys).
- Teacher: zero-shot, the 2B proposed no fixing edit on any seeded bug; the 4B fixed
  diagonal_grasp and tight_verify when asked about the culprit node. SFT data therefore comes
  from loop runs with the 4B as specialist (scripts/gen_repairs.py), on random parameter bugs
  (seeded_bugs.random_bug: same families, other magnitudes, any matching node) on place and
  stack. The fixed BUGS stay the evaluation benchmark; sort stays held out.
- GRPO reward seeds: rehearsal block at offset 20,000 (loop: 0; held-out: 40,000). Cycle-time
  penalty compares successful episodes only. Problems the unedited graph already solves on all
  reward seeds are dropped.
- scripts/train_repair.py wraps mlx-lm-lora. GRPO prompts are rendered with
  enable_thinking=False through a tokenizer proxy, since mlx-lm-lora's GRPO dataset uses the
  template default (thinking on). SFT chat examples already render the no-thinking form.

## 2026-11-24: Phase 6b setup: object variants, camera noise, learned grasp executor
- Why: with the clean sensor, the scripted graphs passed 238/240 dev episodes on the pose and
  distractor splits (180/180 with RGB-D perception). A learned executor had nothing to win, so
  the arbiter result needs failure sources that exist on real robots.
- Object split (model_builder.py, OWNER: done): each cube body gets three extra geoms
  (3 cm box, 5 cm box, upright cylinder 4 cm diameter x 4 cm), off by default (contype 2,
  conaffinity 0, hidden group: nothing collides with them, but they stay in the collision
  tree; geoms compiled with contype 0 are dropped from it and cannot be switched on later). Workcell.reset turns on the variant named by instance.features
  "<cube>_variant" (index into model_builder.VARIANTS; 0 = the 4 cm cube). Body mass and
  inertia stay those of the 4 cm cube (explicit inertial), a stated simplification.
  The sampler draws a variant for every used cube on the object split and spawns it at its
  own half height. success.py uses each object's true half height; the skills keep their
  4 cm assumptions (that is what the split tests).
- Camera noise (Workcell(camera_noise=True), rehearse(camera_noise=...)): the overhead
  RGB-D camera gets a per-episode extrinsic calibration error (rotation N(0, 0.5 deg) per
  axis, translation N(0, 5 mm) per axis, seeded by the instance seed) and Kinect-style depth
  noise (sigma = 2.85e-3 * z^2 m per pixel). Perception back-projects with the calibrated
  (nominal) pose. Off by default so the Phase 1-5 tests and runs keep the clean sensor; the
  Phase 6b and Phase 7 runs turn it on for every method and split, and their configs record it.
- Learned executor: one per node type, starting with grasp (close_gripper nodes). A
  closed-loop behavior-cloned MLP (sklearn, 64x64) that sees the nearest object relative to
  the hand (skills._common.hand_offsets plus 1 mm / 1 deg noise: a privileged stand-in for a
  wrist camera) and outputs a hand-frame velocity plus a close command; a damped-least-squares
  Jacobian step turns the velocity into joint targets. Demonstrator: a privileged
  proportional servo that closes once centered, run from perturbed hand poses (yaw +-0.35 rad,
  xy +-2 cm, z +-8 mm) with DART-style action noise 0.3 so the clone sees recoveries. Close
  commands are oversampled x10 (one per demo). The write-up must state the privileged
  observation.
- Executor semantics (interpreter): "learned" skips the scripted skill's pre-check (that check
  encodes the scripted skill's own assumptions) but keeps its post-check, so both executors
  are judged by the same test. "fallback" runs the skill, then the learned executor from the
  state the failure left. "oracle" snapshots, runs the skill, and on failure restores and runs
  the learned executor (an in-sim upper bound, not deployable).
- CloseGripper.pre now fails when the hand is more than 15 mm off center along the closing
  axis (the open fingers cannot reach both sides). Before, such closes ran and failed their
  post-check; the validity_only policy needs the pre-check to see it.
- Arbiter graphs (arbiter/policies.arbiter_graph): the reference graphs with RGB-D perception
  and verify_grasp's width band widened to [0.02, 0.055] m, so a 3 or 5 cm variant is not
  rejected by the band; a grasp failure is then an executor failure.
- Paired data: at every close_gripper node of an arbiter-graph episode (camera noise on,
  nominal and object splits alternating) both executors run from the same snapshot; the label
  is the node's post-check. The episode continues with the skill.
- Arbiter: one logistic regression per executor on |offsets|, squares, height, yaw, object
  size and roundness; pick learned when its predicted success is higher. Linear on purpose,
  so the coefficients say where each executor wins.
- Rehearsal seed blocks: loop from 0, arbiter paired data 10,000, GRPO rewards 20,000,
  BC demos 30,000, held-out loop checks 40,000.
- run_eval --dev runs the same code on dev seeds (1000 per split) for tuning; only the
  default mode uses eval seeds.
- seeded_bugs.random_bug seeds its generator with "<task>:<seed>" so place and stack get
  different bugs for the same seed.

## 2026-11-24: Arbiter sweep over learned-executor quality
- Why: on dev seeds the 600-demo learned grasp beats the scripted close in every paired state,
  so the learned arbiter always picks it and ties always_learned. That shows the arbiter finds
  the dominant executor, not that per-node choice helps when neither executor dominates.
- What: retrain the behavior-cloned grasp on 25 and 100 demos (same seeds, first N of the
  BC block), refit the arbiter on fresh paired data for each, and run the six policies on the
  nominal and object eval splits. Reported next to the 600-demo rows whatever the outcome.

## 2026-11-24: Evaluation runs every policy with RGB-D perception
- Why: the shared worked example in the agent prompts uses locate_object_gt (ground truth,
  documented as privileged), so LLM-written policies use it, and the camera noise model
  would not touch them while the arbiter graphs face it. A deployed robot has no ground-truth
  locator.
- What: run_eval swaps locate_object_gt for locate_object_rgbd (same ports) in every graph,
  and renames the call in code policies, for all methods and splits. The prompts and the
  Phase 4-6 runs are unchanged.
- Also: run_eval accepts "<graph method>+<policy>" (e.g. graph_loop+learned_arbiter) so the
  full system, an LLM-written and loop-repaired graph run under the learned arbiter, gets
  its own row.
- The loop can rehearse as deployed (run_loop --camera-noise: RGB-D locator swapped in, camera
  noise on in the loop and the held-out check). The graph_loop rows in the evaluation come
  from planner graphs repaired this way; the Phase 5 seeded-bug benchmark keeps the clean
  sensor. rehearsal.runner.with_rgbd is the one place the locator swap lives.

## 2026-11-25: Phase 6 training on the data at hand
- The teacher data run stopped after all place problems and a few stack problems; with the
  first run that gives 48 training and 5 validation SFT examples (deduplicated). That is
  enough for a LoRA on ~40-token JSON edits, so the run was not resumed.
- sft_v2.yaml: sft_v1 with 120 iterations instead of 300 (about 10 epochs instead of 25 at
  batch 4) to limit overfitting on 48 examples; validation every 20 steps.
- Known teacher gap: it never fixes high_verify_min bugs. It picks the right parameter
  (width_min) but sets it to the grasp width as printed in the report (3 decimals), which
  is just above the true width, so the fix fails. SFT cannot learn that family; GRPO's
  sampling can. The report precision is left unchanged so SFT prompts match the data.
- Memory: SFT at batch 4 ran out of GPU memory on 16 GB (Qwen3.5 is a hybrid model, three of
  four layers linear attention, which trains slowly in MLX). sft_v2 and grpo_v1 now process one
  prompt at a time with gradient accumulation (same effective batch) and gradient
  checkpointing; SFT peak memory fell from 12 GB to 3.5 GB at about 15 s per update.
- macOS also aborted a training step with "Impacting Interactivity" (a GPU command buffer ran
  too long). train_repair.py sets MLX_MAX_OPS_PER_BUFFER and MLX_MAX_MB_PER_BUFFER to 8 so
  MLX submits shorter buffers; speed unchanged.

## 2026-11-25: Which authoring rows the results show
- The deployed-condition loops (run_loop --camera-noise) accepted no edit on any task, so
  their final graphs are one more planner call per task, not repaired graphs, and differ from
  graph_single only by planner sampling. The graph_loop and graph_loop+learned_arbiter cells
  stay in results/eval_methods.json, but README and the write-up filter them out
  (methods= on the table marker) and say why.
- The full-system row is graph_multi+learned_arbiter instead: the same per-instruction graphs
  as graph_multi, reused from the first evaluation run (--policies-from), with the learned
  arbiter on the grasp nodes. Reuse keeps the comparison paired.
- Loop results go through report.py like every other number: load_cells turns a run_loop
  metrics.json into two cells (start_graph, after_loop) with split = bug name (or "planner").
  results/loop_bugs.json holds the twelve seeded-bug loops (3 tasks x 4 bugs, 4B specialist,
  clean sensor); results/loop_deployed.json the three deployed-condition loops.

## 2026-11-26: Lint cleanup across done modules
- `ruff check .` (ruff 0.17 default rules) reported 41 findings, mostly in modules finished
  before those rules applied. All fixes keep behavior: sorted imports, literals instead of
  dict()/list(), `X | Y` unions (pydantic treats them like Union), removeprefix, unused noqa
  comments removed.
- runs._git_rev catches only OSError and CalledProcessError (git missing or not a repo).
  test_edit_ops_parse expects pydantic.ValidationError, the error parse_edit documents.
- Kept broad catches where reporting any failure is the point (sanity_check.py diagnostics,
  the headless-GL skip in test_phase0_env), marked with a reasoned noqa.
- RUF012 is ignored for gap/skills: a skill's inputs/outputs/params are class-level
  declarations already typed ClassVar on Skill in base.py; ruff does not see the base class.

## 2026-11-26: A learned grasp that sees through the wrist camera
- Why: the learned grasp reads true hand-object offsets, so it weakly dominates the scripted
  close and the arbiter has nothing to choose (2026-11-24 entries). A learned executor that
  must find the object in its own camera image has failure modes the scripted skill does not
  share. The privileged executor and its results stay as they are.
- Observation, every control step: render the model's wrist camera (on the hand, 6 cm off
  the finger axis, looking down, 75 deg field of view) as RGB-D; segment each cube color with
  the RGB-D locator's HSV mask; back-project through the calibrated hand-eye pose; keep points
  within 5 mm of the highest one (the top face); drop blobs whose top is under 1 cm (the cyan
  zone and the blue floor pass the color mask); take the blob nearest the TCP. From it:
  offsets along the closing and width axes, height, yaw off square, half height (top height /
  2), roundness (top-face fill of its bounding square below 0.9), and `seen` (visible top-face
  area over a full square face). Nothing found: zeros. Same layout as the privileged
  observation plus `seen`.
- Noise (camera_noise on): per-episode hand-eye calibration error, 0.5 deg rotation and 2 mm
  translation per axis (a short rigid mount, so less translation than the overhead camera's
  5 mm), and the same depth noise model. Separate random streams, so the overhead camera's
  noise and all earlier results are unchanged.
- Not modeled: a minimum depth range. Short-range depth cameras lose depth below about 7 cm,
  and at grasp height the top face is 5-6 cm from this camera. Listed as a limitation.
- Demonstrations run with camera noise on, as deployed. The privileged demonstrator still
  chooses the actions; the student is fit on its own wrist observation at each step.
- The arbiter's features come from the same wrist observation (available at run time), plus
  `seen`.
- `train_arbiter.py --obs wrist`. The pickled policy carries its observation mode, so paired
  data, the arbiter and run_eval use the matching observation.
- All parameters above were fixed before any wrist-executor result was seen. Two detector
  bugs were fixed before training (flat blobs: the zone and floor), none after.
- Reporting: results/eval_arbiter_wrist.json (six policies, eval seeds) and
  results/paired_outcomes.json, which report.py builds from each train_arbiter run's
  paired.jsonl (share of paired states where only the skill, only the learned grasp, both,
  or neither passed), so the dominance claim is a generated number like the rest.

## 2026-11-26: Time-aware arbitration (margin sweep)
- Where both grasps succeed the scripted close is faster, so the arbiter gets a margin: it
  picks the learned grasp only if its predicted success beats the skill's by more than the
  margin (Arbiter.choose already takes one). Method name `learned_arbiter@<margin>`.
- Grid fixed in advance from the spread of predicted gaps on the arbiter's own paired
  training states: 0, 0.05, 0.1, 0.2, 0.3, 0.5, plus always_skill and always_learned as the
  two ends. Wrist-camera executor (runs/20261126-002148_train_arbiter), eval seeds, all five
  splits. Every margin is reported; none is picked on eval seeds.
- Time measure: simulated seconds inside grasp nodes per grasp-node visit (sim_steps x dt
  from the trace), so failed episodes that end early do not look fast. Also the share of
  grasp visits run by the scripted skill. run_eval writes both into each cell (grasp_s,
  grasps, skill_grasps); report.py renders them with a `sweep` block.

## 2026-11-26: Invert task: grasps outside the learned executor's training
- Hypothesis, written before any invert result: per-node arbitration beats both fixed
  executors when they fail in different situations. The learned grasp was trained only on
  objects resting on the table, and its wrist estimate assumes it (half height = top height
  / 2). The scripted grasp takes its geometry from the overhead estimate, which handles a cube
  resting on another cube. A dev probe (24 states, red resting on green, place graph,
  wrist executor) agreed: the skill alone succeeded in 8, the learned grasp alone in 1.
- Task `invert`: green starts on red (the stack task's goal state); success is red on green
  (the stack check with roles swapped). Reference graph: grasp green off the stack, put it in
  zone A, grasp red off the table, put it on green. One grasp of each kind per episode, so a
  per-node choice can beat any fixed executor.
- Sampler: task key `start_stacked: {top, bottom}`. After the usual draws (unchanged, so
  other tasks' instances are identical), the top cube takes the bottom cube's xy and rests on
  it (bottom height + its own half height), keeping its own sampled yaw.
- The learned grasp is not retrained: the gap in its training data is what the arbiter must
  route around. Only the arbiter is refit, on paired states from place, stack and invert
  rehearsal seeds (`train_arbiter.py --bc-from` reuses a trained grasp).
- Evaluation: the six policies with the refit arbiter on place, stack and invert (nominal,
  pose, object, distractors) and sort (heldout_task), eval seeds. run_eval gains `--tasks`
  for splits without their own task list.
- Evaluation is deterministic (the 21 cells two runs shared were identical), so on place,
  stack and sort only the refit arbiter is re-run; the fixed policies' rows there stand.
- publish now lets a later run's cell replace an earlier one with the same method, split and
  task (needed after a resumed margin sweep redid part of a split).
