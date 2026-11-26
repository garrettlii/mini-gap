# Progress log

Newest first. One entry per working session: what works (with evidence), what's next, open questions.

## Session 7: stacked tasks and a second try inside the grasp node
- Works: tasks invert and unstack (`start_stacked` takes a list of pairs; unstack's reference
  graph succeeded on 24/24 dev seeds per split probed, nominal and pose, ground-truth
  perception). Arbiter refit on place, stack and invert paired states
  (runs/20261126-020348_train_arbiter). Policies arbiter_fallback and learned_retry
  (interpreter executors fallback_learned, retry_learned). Demo video gains an invert clip.
  `uv run pytest --phase 7` 91 passed; `ruff check .` clean.
- Findings (results/eval_arbiter_invert, eval_arbiter_unstack, eval_arbiter_refit, paired_*,
  retry_*): on invert the scripted close wins some stacked-grasp states and the refit arbiter
  beats both single executors episode by episode, with faster grasps; on unstack, with no
  unstack data, it transfers. A second try inside the grasp node is the bigger effect: the
  learned grasp retried beats always_learned on every task; switching executor instead adds
  to that on unstack (pre-registered), not on the table tasks, not significantly on invert.
- Decisions: DECISIONS 2026-11-26 entries invert, unstack (with the reset finding),
  arbiter_fallback, learned_retry.
- Next: a settle tolerance that grows with stack height; fail edges from grasp nodes in the
  reference graphs, now that a retry is known to help; an arbiter trained on outcomes with
  the second try (pick the first executor to maximize success after the fallback).
- Open: some unstack object-split towers fail reset and count against every policy;
  arbiter_fallback was designed after seeing invert's first rows (unstack is its clean test).

## Session 6: wrist-camera learned grasp
- Works: gap/arbiter/wrist.py, a learned grasp that finds the object in the wrist RGB-D image
  (hand-eye calibration error and depth noise under the noise model); train_arbiter --obs
  wrist in runs/20261126-002148_train_arbiter, eval in runs/20261126-003116_eval.
  `uv run pytest --phase 7` 83 passed; `ruff check .` clean.
- Finding: the wrist grasp is weaker than the privileged one but its failures nest inside the
  scripted close's (results/paired_outcomes.json: skill-only states 3/392), so the learned
  arbiter again ties always_learned on every split (results/eval_arbiter_wrist.json).
- Decisions: DECISIONS 2026-11-26 wrist entry (design fixed before results), lint cleanup.
- Next: an arbiter that weighs cycle time as well as success (the scripted close is faster
  where both succeed; make_policy already takes a margin).
- Open: no minimum depth range in the wrist camera model.

## Session 5: Phase 5-7 results published
- Works: results/ now holds every headline number (eval_arbiter, eval_arbiter_bc25/bc100,
  repair, eval_methods, loop_bugs, loop_deployed); README and docs/WRITEUP.md render from
  them (`report.py --render`). `uv run pytest --phase 7` 80 passed. Demo video in
  docs/media/demo.mp4.
- Phase 5: seeded-bug loops (3 tasks x 4 bugs, 4B specialist) raise held-out success from
  0/64 to 64/64 on grasp yaw and depth bugs for place and stack (results/loop_bugs.json);
  tight_verify fixed only on sort, no_release never. Over 330 edit records.
- Phase 6: repair model zero-shot < SFT <= GRPO on all three tasks, sort held out
  (results/repair.json); GRPO learned to restore default values; no high_verify_min fixes.
- Phase 6b/7: learned arbiter ties always_learned on every split (weak dominance, also at 25
  and 100 demos). Authoring methods: all fail stack and sort; critic edits change no outcome;
  code's place release bug; the arbiter lifts LLM place graphs except on the object split.
- Decisions: DECISIONS 2026-11-25 "Which authoring rows the results show" (graph_loop rows
  filtered out: the deployed loop accepted no edit).
- Next: optional: a learned executor with its own failure modes (wrist-image detection) to
  give the arbiter something to arbitrate; higher-temperature GRPO; making the repo public.
- Open: 41 pre-existing ruff findings across older modules (`ruff check .`); report widths
  print at 3 decimals (blocks high_verify_min repairs).

## Session 4: Phases 5-6 code, Phase 6b arbiter, Phase 7 tooling
- Works: `uv run pytest --phase 7` 78+ passed (executor semantics, arbiter, object split,
  camera noise, run_eval, report rendering). Repair loop with routed specialist/critic;
  teacher repair data in runs/20261124-200353_gen_repairs and runs/20261124-224356_gen_repairs
  (330+ edit records, 48 train / 5 valid SFT examples). Learned grasp (600 demos) and arbiter in
  runs/20261124-230951_train_arbiter; dev comparison of the six policies in
  runs/20261124-231427_eval; eval-seed arbiter run (nominal + pose done) in
  runs/20261124-232414_eval.
- Finding: the learned grasp weakly dominates the scripted close (in the paired data no state
  has only the skill succeeding, even with 25 demos), so the learned arbiter ties
  always_learned. Logged before the executor-quality sweep (DECISIONS 2026-11-24).
- Decisions: DECISIONS entries for Phase 6b setup, executor-quality sweep, RGB-D perception
  at evaluation, loop under deployment conditions, Phase 6 training on the data at hand.
- Next: the queued GPU chain (SFT v2, GRPO, eval_repair, deployed-condition planner loops,
  methods eval, seeded-bug benchmark loops) and CPU chain (remaining arbiter splits, sweep);
  then publish results/, render README and WRITEUP, record videos.
- Open: report prints widths to 3 decimals (blocks high_verify_min repairs); per-node oracle
  is not an episode upper bound; long runs must survive session ends (queues now detached
  with done markers under runs/_queue).

## Session 3: Phases 1-3 done, Phase 4 harness built
- Works: `uv run pytest --phase 4 --runslow` covers phases 0-4. Oracle place 100/100 dev seeds;
  place graph 100/100 (within 2 pts of the oracle). Example graphs on 50 dev seeds: place 50/50,
  stack 50/50, sort 48/50. RGBD locator 0.8 mm mean error. Phase 4 harness tests (scripted
  backend) pass; local Qwen planner calls take ~50 s each on this Mac.
- Decisions: DECISIONS.md 2026-11-24 entries (local LLM only, IK restarts, gravity compensation,
  stiffer gripper, zone slots, adaptive settle, Phase 4 methods and worked example).
- Next: run scripts/first_shot.py for the Phase 4 table; then Phase 5 rehearsal loop.
- Open: 2B planner first-shot quality unknown; frontier rows dropped (no API spend).

## Session 2: Workcell.get_state, task_sampler.sample, Workcell.reset
- Works: `uv run pytest --phase 1 -k "state_has_contract_keys or reset_places_objects or
  reset_rejects or sampler"` 4 passed; phase 0 still 9 passed. Sampler is identical across
  processes (sha256 of 200 instances equal under 3 PYTHONHASHSEEDs). A 200-reset dev sweep
  (4 tasks x 50 seeds) passes the 1 mm settle check. New test: reset raises on overlapping cubes.
- Decisions: contacts are a sorted list of sorted body-name pairs. Yaw feature folded to
  [-pi/4, pi/4) (raw kept as *_raw_yaw). Unused cubes park at (0.90, 0.60/0.45/-0.60), on the
  table and beyond reach. reset() raises rather than repairs a bad spawn (keeps seed -> scene).
- Next: step(ctrl, n), snapshot/restore (test_snapshot_restore_is_exact), render; then IK.
- Open: min_separation 0.07 is too tight for the Phase 7 5 cm cube variant (corners overlap at
  45 deg). ruff nits in workcell.py / task_sampler.py (import order, dict() literals).

## Session 1: Phase 0 done, Phase 1 kickoff
- Works: setup_mac.sh re-runs clean; `uv run pytest` 9 passed / 30 skipped (phase-gated);
  sanity_check all ok except ".env has an API key" (key not added yet); runs/sanity/cameras.png
  renders all three cameras. `mlx_lm.generate` answers with mlx-community/Qwen3.5-2B-4bit.
  configs/models.yaml: local 2B-4bit.
- Model notes: bodies/geoms/joints/sites, affine position servos (ctrl = target angle),
  gripper tendon 0-255 remap, home keyframe, freejoint qpos 7 vs qvel 6,
  nv = 9 + 3*6, green cube slices qpos[16:23] and qvel[15:21].
- Next: check what ctrl[3] = -1.0 means and the width at ctrl[7] = 128. Then get_state (check in a
  REPL first), task_sampler.sample, reset; target
  `uv run pytest --phase 1 -k "state_has_contract_keys or reset_places_objects"`.
- Open: Qwen3.5 answers in thinking mode by default; MLXBackend must turn it off (verify the
  chat-template flag). 4-bit vs bf16 base for Phase 6 LoRA. docs/BUILD_GUIDE.md missing.
  sanity_check.py always exits 0. Park unused cubes ON the table, out of reach (not off the edge).

## Session 0: scaffold
- Works: workcell.xml compiles against Menagerie Panda (nq 30, nu 8); cubes settle; three
  cameras render offscreen; phase-0 tests pass.
- Next: run scripts/setup_mac.sh on the Mac; fill configs/models.yaml; start Phase 1 (get_state).
- Open: which mlx-community Qwen3.5 repo fits this Mac's RAM.
