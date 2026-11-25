# Mini-GaP: learned per-node arbitration and small-model repair for graph-as-policy robots

Every number in this document is generated from results/*.json by
`uv run python -m gap.eval.report --render docs/WRITEUP.md`. Rates are success fractions
with Wilson 95% intervals and raw counts.

## Abstract

Graph-as-Policy (GaP) systems have language models write robot policies as graphs of typed
skills, rehearse them in simulation, and repair the node that failed. Each node runs a
scripted skill. Recent agentic manipulation work names, as an open problem, letting each node
choose at run time between its scripted skill and a learned policy. We build a laptop-scale
GaP in MuJoCo with a Franka Panda, add a learned grasp executor, and train a per-node arbiter
from paired executor outcomes collected from identical simulator states. We compare six
arbitration policies across five evaluation splits with a realistic overhead-camera noise
model. Separately, we train a 2B-parameter Qwen model with supervised fine-tuning and GRPO on
simulator rewards to repair broken graphs, and test it on a task held out from training. All
models run locally; no paid API is used.

## 1 Introduction

Writing a robot policy as a graph of skills makes failures local. When a run fails, the trace
names the node, its pre- and post-checks, and the state around it, so a repair can target one
parameter of one node instead of rewriting the program. GaP showed that this structure lets
language-model agents improve policies through rehearsal in simulation.

Scripted skills are fast and predictable inside the conditions their author assumed, and they
fail outside them. A top-down grasp that trusts a perception estimate closes on air when the
estimate is a centimeter off. Learned closed-loop policies can absorb that error, but they are
slower and their failures are harder to predict. A node does not have to commit to one
executor. This project asks when a learned arbiter should hand a node to the learned policy,
and how close that choice comes to the best executor picked in hindsight.

The second question is whether a small local model can do the repair work. The repair loop
asks a node specialist for one parameter edit given a failure report. We train Qwen3.5-2B on
edits a larger local model found, then on simulator rewards, and compare it with the
untrained model on trained tasks and on a held-out task.

Contributions:
1. A complete, reproducible laptop-scale GaP stack: skill library with checks, graph schema
   and interpreter, parallel rehearsal, per-node diagnosis, routed repair loop.
2. A learned per-node arbiter between a scripted skill and a behavior-cloned policy, trained
   on paired outcomes from identical states, compared with five other arbitration policies.
3. SFT and GRPO training of a 2B repair model on simulator rewards, with a held-out task.

## 2 System

**Workcell.** MuJoCo with the Menagerie Franka Panda, a table with colored cubes and two
goal zones, and three cameras. Tasks: place (cube into a zone), stack (green on red), and
sort (three cubes into two zones by color). Each task instance is fully determined by
(task, split, seed).

**Skills.** Ten typed skills with declared inputs, outputs, tunable parameters with ranges,
and pre/post checks: grasp planning, close/open gripper, grasp verification, three motion
skills, ground-truth and RGB-D object localization, and place-pose computation. Motion uses
damped-least-squares inverse kinematics.

**Graphs.** A policy is a JSON graph of nodes (skill, inputs, parameter overrides). Success
moves to the next node or an explicit success edge; failure follows a fail edge with a retry
budget or ends the episode. Inputs may reference earlier nodes' outputs. A validator rejects
unknown skills, bad references, out-of-range parameters, and unbounded loops before anything
runs. Task success is judged only by the simulator's success check, never by the graph.

**Rehearsal and diagnosis.** A graph is rehearsed on a batch of seeded instances in parallel
worker processes. Diagnosis blames the first node whose check failed in each failing episode,
groups episodes by blamed node and failure category, and reports numeric patterns (for
example, failures concentrated at large object yaw) and, for grasp failures, the measured
hand-object geometry.

**Repair loop.** The loop routes the most-blamed failure: parameter and skill problems go to
a node specialist for the failing node and for upstream suspects along the data flow;
ordering and missing-step problems go to a structure critic. Every proposed edit is
validated, rehearsed on the same instances, and kept only if success improves. Rejected edits
are shown back to the specialist so it does not repeat them. Seeds are split into disjoint
blocks: rehearsal seeds for the loop, training data and the arbiter; dev seeds for tuning;
and fixed per-split evaluation seeds used only by the evaluation runner.

## 3 Learned per-node arbitration

**Failure sources.** With a perfect sensor the scripted graphs succeed on almost every
instance, so a learned executor has nothing to win. We therefore add two failure sources real
cells have, applied identically to every method:
- *Overhead camera noise*: a per-episode extrinsic calibration error (rotation 0.5° and
  translation 5 mm standard deviation per axis) plus depth noise growing with the square of
  the distance, as in structured-light sensors. Perception back-projects with the nominal
  calibration, so its estimates are off by several millimeters to a couple of centimeters.
- *Object variants* (object split): each cube may be replaced by a 3 cm cube, a 5 cm cube, or
  an upright cylinder, while the skills keep their 4 cm assumptions. Variants keep the
  original mass and inertia.

**Executors.** Each node runs one of: the scripted skill; the learned executor (which skips
the skill's pre-check, since that check encodes the scripted skill's assumptions, but must
pass the same post-check); *fallback* (skill, then the learned executor from wherever the
failure left the scene); or *oracle* (skill; on failure, restore the snapshot and run the
learned executor), an in-simulation upper bound no real robot can run.

**Learned grasp.** A closed-loop MLP behavior-cloned from a privileged servo demonstrator. It
observes the nearest object's position and yaw relative to the hand, with millimeter noise,
plus gripper width and height. This is a stand-in for a wrist camera, and it is privileged:
the policy does not have to detect the object. It outputs a hand-frame velocity and a close
command; a damped-least-squares Jacobian step converts the velocity into joint targets.
Demonstrations start from perturbed hand poses (±2 cm, ±0.35 rad) with DART-style action noise.

**Paired data and the arbiter.** At every grasp node of rehearsal episodes we snapshot the
simulator, run the scripted skill, restore, and run the learned executor, recording both
post-check outcomes and the hand-object features at the snapshot. Pairs from identical states
separate "which executor" from "which situation". The arbiter is one logistic regression per
executor on the absolute offsets, their squares, height, yaw, object size and roundness; it
picks the executor with the higher predicted success. Linear models keep the decision
readable from the coefficients.

**Policies compared.** always_skill; always_learned; validity_only (skill unless its
pre-check fails, the gating used in prior agentic manipulation work); fallback_on_failure;
learned_arbiter; oracle.

## 4 Training a small repair model

**Data.** Random parameter bugs (grasp yaw offsets, grasp depth, verification bands) are
injected into the reference graphs for place and stack, and the repair loop runs with the
larger local model (Qwen3.5-4B) as the specialist. Accepted, improving edits become SFT
examples in exactly the prompt format the specialist sees at run time. Every specialist query
becomes a GRPO prompt; prompts whose graph already succeeds on all reward seeds are dropped.

**Reward.** Zero for unparseable output, 0.05 for a parseable but invalid edit, otherwise 0.1
plus the change in success over eight rehearsal seeds, minus 0.2 if the mean cycle time of
successful episodes grows by more than half. Rehearsals are cached by graph.

**Training.** LoRA on Qwen3.5-2B (4-bit) with mlx-lm-lora: SFT, then GRPO from the SFT
adapter, with thinking disabled to match deployment.

**Evaluation.** Held-out random bugs (bug seeds disjoint from training) on place and stack
(trained) and sort (held out). Each model gets the same buggy graph, node, and diagnosis
report, proposes one edit with up to three validator-feedback retries, and the edited graph is
rehearsed on held-out rehearsal seeds.

## 5 Results

### 5.1 Arbitration (evaluation seeds, camera noise on)

Six policies on the arbiter graphs (reference graphs with RGB-D perception and a verification
band that admits every object variant). The instruction split does not apply: these graphs
are fixed and never read the instruction.

<!-- table: eval_arbiter -->
| method | nominal | pose | object | distractors | heldout_task |
|---|---|---|---|---|---|
| always_skill | 0.42 [0.36, 0.49] (85/200) | 0.50 [0.43, 0.57] (100/200) | 0.46 [0.39, 0.52] (91/200) | 0.48 [0.42, 0.55] (97/200) | 0.37 [0.28, 0.47] (37/100) |
| always_learned | 0.92 [0.87, 0.95] (184/200) | 0.94 [0.89, 0.96] (187/200) | 0.83 [0.77, 0.88] (166/200) | 0.95 [0.92, 0.98] (191/200) | 0.81 [0.72, 0.87] (81/100) |
| validity_only | 0.55 [0.48, 0.61] (109/200) | 0.63 [0.56, 0.69] (126/200) | 0.55 [0.48, 0.62] (110/200) | 0.60 [0.54, 0.67] (121/200) | 0.48 [0.38, 0.58] (48/100) |
| fallback_on_failure | 0.76 [0.70, 0.81] (152/200) | 0.80 [0.73, 0.85] (159/200) | 0.69 [0.62, 0.75] (137/200) | 0.86 [0.81, 0.90] (172/200) | 0.66 [0.56, 0.75] (66/100) |
| learned_arbiter | 0.92 [0.87, 0.95] (184/200) | 0.94 [0.89, 0.96] (187/200) | 0.83 [0.77, 0.88] (166/200) | 0.95 [0.92, 0.98] (191/200) | 0.81 [0.72, 0.87] (81/100) |
| oracle | 0.88 [0.82, 0.91] (175/200) | 0.90 [0.84, 0.93] (179/200) | 0.77 [0.70, 0.82] (153/200) | 0.94 [0.90, 0.97] (188/200) | 0.81 [0.72, 0.87] (81/100) |
<!-- /table -->

What the table shows:
- Gating by the scripted skill's own pre-check (validity_only) closes only a minority of the
  gap between the scripted and the learned grasp: most failed scripted grasps pass the
  pre-check and fail afterwards, where a pre-check cannot see them.
- Falling back to the learned policy after a failed grasp helps, but less than choosing it
  up front, likely because a failed close leaves the object pushed or tilted.
- The learned arbiter's row equals always_learned on every split. On its own paired training
  states it predicts higher success for the learned grasp in all of them, and the data say
  why: the learned grasp re-centers the hand before closing, and from a centered start its
  first action is the same close the scripted skill performs. In the paired data there is no
  state where only the scripted close succeeded. The learned executor weakly dominates, so
  the arbiter learns to defer to it; it adds nothing over a fixed always-learned rule here.
- The per-node oracle falls below always_learned on most splits: it keeps the scripted grasp
  whenever that grasp passes its post-check, and some of those grasps slip later.

Learned arbiter vs validity-only on the object split:
<!-- rate: eval_arbiter learned_arbiter object -->0.83 [0.77, 0.88] (166/200)<!-- /rate --> vs <!-- rate: eval_arbiter validity_only object -->0.55 [0.48, 0.62] (110/200)<!-- /rate -->.

**Executor-quality sweep.** The same policies with the learned grasp cloned from 25 and 100
demonstrations (arbiter refit for each), on the nominal and object splits.

<!-- table: eval_arbiter_bc25 -->
| method | nominal | object |
|---|---|---|
| always_skill | 0.42 [0.36, 0.49] (85/200) | 0.46 [0.39, 0.52] (91/200) |
| always_learned | 0.87 [0.82, 0.91] (174/200) | 0.80 [0.73, 0.85] (159/200) |
| validity_only | 0.54 [0.47, 0.60] (107/200) | 0.52 [0.45, 0.59] (104/200) |
| fallback_on_failure | 0.64 [0.57, 0.70] (127/200) | 0.61 [0.54, 0.67] (122/200) |
| learned_arbiter | 0.87 [0.82, 0.91] (174/200) | 0.80 [0.73, 0.85] (159/200) |
| oracle | 0.83 [0.77, 0.88] (166/200) | 0.74 [0.68, 0.80] (148/200) |
<!-- /table -->

<!-- table: eval_arbiter_bc100 -->
| method | nominal | object |
|---|---|---|
| always_skill | 0.42 [0.36, 0.49] (85/200) | 0.46 [0.39, 0.52] (91/200) |
| always_learned | 0.91 [0.86, 0.94] (182/200) | 0.83 [0.77, 0.88] (166/200) |
| validity_only | 0.54 [0.47, 0.61] (108/200) | 0.56 [0.49, 0.62] (111/200) |
| fallback_on_failure | 0.64 [0.57, 0.70] (128/200) | 0.66 [0.59, 0.72] (132/200) |
| learned_arbiter | 0.91 [0.86, 0.94] (182/200) | 0.83 [0.77, 0.88] (166/200) |
| oracle | 0.87 [0.82, 0.91] (174/200) | 0.77 [0.71, 0.82] (154/200) |
<!-- /table -->

With fewer demonstrations the learned grasp is weaker, but from the same state it still
never fails where the scripted close succeeds, and the refit arbiters again pick it in all or
nearly all paired states. Per-node arbitration pays off when the executors fail in different
situations. A learned executor that reduces to the scripted action whenever the scripted
assumptions hold does not create such situations; a learned policy with its own failure modes
(for example one that must detect the object from a wrist image) would be the test case.

### 5.2 Policy authoring methods (evaluation seeds, camera noise on)

Free-form code over the same skills, a single planner call, and planner plus structure
critic, each written once per (task, instruction) by the local 4B model.

<!-- table: eval_methods -->
<!-- /table -->

### 5.3 Repair models

Fraction of held-out bugs where the model's edit raised success.

<!-- table: repair by=task -->
<!-- /table -->

## 6 Limitations

- Simulation only. The camera noise model and object variants were added by us so that a
  learned executor has failures to fix; they are applied equally to all methods.
- The learned grasp's observation is privileged (true relative pose plus small noise). A real
  wrist camera would need detection and would add its own failures.
- Object variants keep the 4 cm cube's mass and inertia.
- Only the grasp node type has a learned executor; reach and place nodes always run their
  skills.
- The oracle executor restores a simulator snapshot and is not deployable. It is a per-node
  bound, not a per-episode one: it keeps the skill's result whenever the skill passes its
  post-check, even if that grasp later slips, so an always-learned policy can beat it.
- All language models are small local Qwen models; we report no frontier-model rows.

## 7 Related work

GaP (arXiv 2607.05369) introduced graph-structured robot policies written and repaired by
language-model agents with simulator rehearsal. Code-as-policies systems such as CaP-X write
free-form programs over skill APIs; our code baseline follows that design with the same
skills. AW-0 (arXiv 2610.30136) gates scripted skills by validity checks and names learned
per-node arbitration as future work, which Section 3 addresses.

## Reproduce

```bash
./scripts/setup_mac.sh
uv run python scripts/train_arbiter.py --demos 600 --paired 200
uv run python -m gap.eval.run_eval --models runs/<train_arbiter> \
    --methods always_skill always_learned validity_only fallback_on_failure learned_arbiter oracle \
    --splits nominal pose object distractors heldout_task
uv run python -m gap.eval.report --publish eval_arbiter runs/<eval>
uv run python -m gap.eval.report --render README.md docs/WRITEUP.md
```
