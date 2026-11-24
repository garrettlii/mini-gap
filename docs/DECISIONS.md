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
