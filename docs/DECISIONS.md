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
