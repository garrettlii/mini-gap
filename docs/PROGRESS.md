# Progress log

Newest first. One entry per working session: what works (with evidence), what's next, open questions.

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
