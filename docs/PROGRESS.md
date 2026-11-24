# Progress log

Newest first. One entry per working session: what works (with evidence), what's next, open questions.

## Session 3: Phase 1 done
- Works: `uv run pytest --phase 1 --runslow` 18 passed; oracle place 100/100 dev seeds
  (Wilson [0.96, 1.00]). IK converges on all 20 test targets incl. wrist-limit cases.
  `scripts/record_oracle.py <seeds>` saves MP4s into a RunDir.
- Decisions: see DECISIONS.md 2026-11-24 (local LLM only, Phase 1 choices).
  Private GitHub remote created (garrettlii/mini-gap).
- Next: Phase 2 skills (oracle rewritten as skill calls, >= 95%).
- Open: frontier rows dropped (no API spend). Pre-existing ruff nits in workcell/task_sampler.

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
