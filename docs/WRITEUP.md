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

In our setting the learned grasp almost never fails from a state where the scripted grasp
succeeds, whether it reads the true object offset or has to find the object in its own
wrist-camera image. The learned arbiter therefore reduces to choosing the learned grasp: it
ties always-learned and does not fall below the per-node oracle. Where both succeed the
scripted grasp is faster, so a useful arbiter would weigh cycle time as well as success. The trained repair model fixes more held-out bugs than the untrained
one on every task, including the held-out task. The rehearsal loop repairs seeded
grasp-geometry bugs on every task, but the local 4B model's first-draft graphs fail stack and
sort, and single edits do not repair them under camera noise.

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
sort (three cubes into two zones by color). Two later tasks start from a stack, so a grasp
must take a cube off another cube: invert (green starts on red; red must end on green) and
unstack (a blue-green-red tower, sorted into the zones). Each task instance is fully
determined by (task, split, seed).

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

**Learned grasp from the wrist camera.** A second learned grasp has to find the object
itself. At every control step it renders the hand's RGB-D camera (6 cm off the finger axis,
looking down, 75° field of view), segments each cube color, back-projects the pixels through
the calibrated hand-eye pose, keeps the top face, and takes the blob nearest the fingers.
From that it computes the same quantities as the privileged observation, plus the visible
fraction of the top face. Under the noise model the hand-eye calibration carries its own
per-episode error (0.5° and 2 mm per axis) and the same depth noise. Near the object the top
face can leave the image or hide behind a finger, and then the estimate is wrong in ways the
scripted skill, which trusts the overhead estimate, does not share. The demonstrator still
acts on the true offsets; the student is fit on its wrist observation. The arbiter for this
executor sees the same wrist observation, which is available at run time. Every parameter
was fixed before any result was seen.

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
is the test case, which the wrist-camera grasp provides.

**Learned grasp from the wrist camera.** The same six policies with the wrist-camera grasp
(Section 3) and its refit arbiter, on the same evaluation seeds.

<!-- table: eval_arbiter_wrist -->
| method | nominal | pose | object | distractors | heldout_task |
|---|---|---|---|---|---|
| always_skill | 0.42 [0.36, 0.49] (85/200) | 0.50 [0.43, 0.57] (100/200) | 0.46 [0.39, 0.52] (91/200) | 0.48 [0.42, 0.55] (97/200) | 0.37 [0.28, 0.47] (37/100) |
| always_learned | 0.75 [0.69, 0.80] (150/200) | 0.84 [0.79, 0.89] (169/200) | 0.78 [0.71, 0.83] (155/200) | 0.80 [0.74, 0.85] (160/200) | 0.72 [0.63, 0.80] (72/100) |
| validity_only | 0.53 [0.46, 0.59] (105/200) | 0.62 [0.56, 0.69] (125/200) | 0.55 [0.48, 0.61] (109/200) | 0.59 [0.52, 0.66] (118/200) | 0.48 [0.38, 0.58] (48/100) |
| fallback_on_failure | 0.64 [0.57, 0.70] (127/200) | 0.73 [0.67, 0.79] (147/200) | 0.66 [0.59, 0.72] (131/200) | 0.72 [0.65, 0.78] (144/200) | 0.59 [0.49, 0.68] (59/100) |
| learned_arbiter | 0.74 [0.68, 0.80] (149/200) | 0.84 [0.79, 0.89] (169/200) | 0.78 [0.72, 0.83] (156/200) | 0.81 [0.74, 0.85] (161/200) | 0.70 [0.60, 0.78] (70/100) |
| oracle | 0.73 [0.66, 0.79] (146/200) | 0.81 [0.76, 0.86] (163/200) | 0.72 [0.66, 0.78] (145/200) | 0.81 [0.74, 0.85] (161/200) | 0.70 [0.60, 0.78] (70/100) |
<!-- /table -->

**Paired outcomes.** For each learned grasp, the share of paired training states (place and
stack, nominal and object splits, camera noise on) by which executors passed the node's
post-check from the same snapshot. Columns: observation and number of demonstrations.

<!-- table: paired_outcomes -->
| method | privileged/600 | privileged/25 | privileged/100 | wrist/600 |
|---|---|---|---|---|
| skill_only | 0.00 [0.00, 0.01] (0/392) | 0.00 [0.00, 0.01] (0/392) | 0.00 [0.00, 0.01] (0/392) | 0.01 [0.00, 0.02] (3/392) |
| learned_only | 0.47 [0.42, 0.52] (184/392) | 0.39 [0.35, 0.44] (154/392) | 0.46 [0.41, 0.51] (179/392) | 0.33 [0.29, 0.38] (131/392) |
| both | 0.52 [0.47, 0.57] (203/392) | 0.52 [0.47, 0.57] (203/392) | 0.52 [0.47, 0.57] (203/392) | 0.51 [0.46, 0.56] (200/392) |
| neither | 0.01 [0.01, 0.03] (5/392) | 0.09 [0.06, 0.12] (35/392) | 0.03 [0.01, 0.05] (10/392) | 0.15 [0.12, 0.19] (58/392) |
<!-- /table -->

The wrist camera costs the learned grasp success (always_learned on nominal:
<!-- rate: eval_arbiter always_learned nominal -->0.92 [0.87, 0.95] (184/200)<!-- /rate --> reading the true offset,
<!-- rate: eval_arbiter_wrist always_learned nominal -->0.75 [0.69, 0.80] (150/200)<!-- /rate --> from the wrist
image), and the share of paired states where neither executor succeeds rises from
<!-- rate: paired_outcomes neither privileged/600 -->0.01 [0.01, 0.03] (5/392)<!-- /rate --> to
<!-- rate: paired_outcomes neither wrist/600 -->0.15 [0.12, 0.19] (58/392)<!-- /rate -->. But the new failures fall
almost entirely on states where the scripted close fails as well: the scripted close alone
succeeds in <!-- rate: paired_outcomes skill_only wrist/600 -->0.01 [0.00, 0.02] (3/392)<!-- /rate --> of paired
states. Those states are too few for the refit arbiter to find a region where the skill is
better, so it still sends almost every grasp to the learned policy, and its row matches
always_learned within a few episodes on every split. The shared failures are not simply the
far-off starts: in the paired data they come with smaller closing-axis offsets than the
states only the learned grasp wins, and with less of the top face in view.

So in this workcell the scripted close almost never wins on success, even against a learned grasp
that must find the object itself. Its advantage is time: it closes at once, while the learned
grasp servos first.

**Time-aware arbitration.** The arbiter picks the learned grasp only when its predicted
success beats the skill's by more than a margin, so a larger margin hands more grasps to the
faster skill. Wrist-camera executor, all five splits pooled; the margin grid was fixed in
advance from the arbiter's training data and every margin is shown. Grasp time is simulated
seconds inside grasp nodes per grasp-node visit.

<!-- sweep: arbiter_margin -->
| method | success | grasps by skill | s per grasp |
|---|---|---|---|
| always_skill | 0.46 [0.42, 0.49] (410/900) | 1.00 [1.00, 1.00] (963/963) | 0.26 |
| always_learned | 0.78 [0.76, 0.81] (706/900) | 0.00 [0.00, 0.00] (0/1013) | 0.70 |
| learned_arbiter | 0.78 [0.76, 0.81] (705/900) | 0.04 [0.03, 0.05] (37/1012) | 0.70 |
| learned_arbiter@0.05 | 0.78 [0.75, 0.81] (703/900) | 0.11 [0.09, 0.13] (112/1012) | 0.68 |
| learned_arbiter@0.1 | 0.77 [0.74, 0.80] (692/900) | 0.19 [0.16, 0.21] (189/1009) | 0.66 |
| learned_arbiter@0.2 | 0.74 [0.71, 0.76] (663/900) | 0.33 [0.30, 0.36] (333/1008) | 0.60 |
| learned_arbiter@0.3 | 0.69 [0.66, 0.72] (620/900) | 0.49 [0.45, 0.52] (486/1001) | 0.52 |
| learned_arbiter@0.5 | 0.61 [0.58, 0.64] (550/900) | 0.70 [0.68, 0.73] (696/988) | 0.44 |
<!-- /sweep -->

Small margins (0.05 and 0.1) hand a tenth to a fifth of the grasps to the scripted close
while pooled success stays inside always_learned's interval. Grasp time falls only a little,
because the handed-over grasps are a minority and each saves only the gap between the two
grasp times in the table. Larger margins buy more time at a steady cost in success. No margin
beats always_learned on success, as expected when the skill alone almost never wins: on these
tasks the margin is a speed dial, not a way to raise success.

**Grasping off a stack.** Place, stack and sort start with every cube on the table, the
states the learned grasp was trained on. Invert starts with green on red, so its first grasp
takes green off red: no demonstration starts there, and the wrist estimate of the object's
half height (top height over two, which assumes the object rests on the table) is off by a
whole cube. We refit the arbiter on paired states from place, stack and invert (rehearsal
seeds, the same wrist-camera learned grasp). Invert is the first task with more than a
handful of states only the scripted close wins (share of paired states per task):

<!-- table: paired_outcomes_refit by=task -->
| method | place | stack | invert |
|---|---|---|---|
| skill_only | 0.01 [0.00, 0.04] (2/195) | 0.01 [0.00, 0.03] (1/197) | 0.05 [0.03, 0.08] (18/339) |
| learned_only | 0.31 [0.25, 0.38] (61/195) | 0.36 [0.29, 0.42] (70/197) | 0.17 [0.13, 0.21] (57/339) |
| both | 0.53 [0.46, 0.60] (103/195) | 0.49 [0.42, 0.56] (97/197) | 0.65 [0.60, 0.70] (222/339) |
| neither | 0.15 [0.11, 0.21] (29/195) | 0.15 [0.10, 0.20] (29/197) | 0.12 [0.09, 0.16] (42/339) |
<!-- /table -->

Invert, evaluation seeds, camera noise on, the refit arbiter:

<!-- table: eval_arbiter_invert -->
| method | nominal | pose | object | distractors |
|---|---|---|---|---|
| always_skill | 0.35 [0.26, 0.45] (35/100) | 0.46 [0.37, 0.56] (46/100) | 0.40 [0.31, 0.50] (40/100) | 0.41 [0.32, 0.51] (41/100) |
| always_learned | 0.43 [0.34, 0.53] (43/100) | 0.56 [0.46, 0.65] (56/100) | 0.61 [0.51, 0.70] (61/100) | 0.53 [0.43, 0.62] (53/100) |
| validity_only | 0.41 [0.32, 0.51] (41/100) | 0.52 [0.42, 0.62] (52/100) | 0.44 [0.35, 0.54] (44/100) | 0.48 [0.38, 0.58] (48/100) |
| fallback_on_failure | 0.56 [0.46, 0.65] (56/100) | 0.65 [0.55, 0.74] (65/100) | 0.50 [0.40, 0.60] (50/100) | 0.57 [0.47, 0.66] (57/100) |
| learned_arbiter | 0.49 [0.39, 0.59] (49/100) | 0.61 [0.51, 0.70] (61/100) | 0.63 [0.53, 0.72] (63/100) | 0.57 [0.47, 0.66] (57/100) |
| oracle | 0.48 [0.38, 0.58] (48/100) | 0.65 [0.55, 0.74] (65/100) | 0.59 [0.49, 0.68] (59/100) | 0.57 [0.47, 0.66] (57/100) |
<!-- /table -->

Matched episode by episode (same split and seed; the sign test counts the episodes only one
of the two policies solved):

<!-- paired: paired_invert -->
| A vs B | episodes | only A succeeded | only B succeeded | sign test p |
|---|---|---|---|---|
| learned_arbiter vs always_learned | 400 | 22 | 5 | 0.0015 |
| learned_arbiter vs always_skill | 400 | 89 | 21 | 3.8e-11 |
| learned_arbiter vs fallback_on_failure | 400 | 53 | 51 | 0.92 |
<!-- /paired -->

Here the arbiter beats both single executors. In a dev-seed check, the grasps it gives the
scripted close are almost all the stacked one, and in the evaluation episodes it has fewer
failures at the stacked grasp than always_learned. It
does not beat fallback_on_failure, which runs the skill and, after a failed grasp, the
learned one within the same node visit: the two solve different episodes, and the counts
are close to even.

The rule costs nothing where it does not apply: on place, stack and sort (all five splits),
the refit arbiter still matches always_learned episode by episode:

<!-- paired: paired_refit -->
| A vs B | episodes | only A succeeded | only B succeeded | sign test p |
|---|---|---|---|---|
| learned_arbiter vs always_learned | 900 | 5 | 7 | 0.77 |
<!-- /paired -->

### 5.2 Policy authoring methods (evaluation seeds, camera noise on)

Free-form code over the same skills, a single planner call, and planner plus structure
critic, each written once per (task, instruction) by the local 4B model. The last row runs
the graph_multi graphs unchanged except that the learned arbiter picks each grasp node's
executor. The instruction split uses paraphrased instructions.

<!-- table: eval_methods methods=code,graph_single,graph_multi,graph_multi+learned_arbiter -->
| method | nominal | pose | object | instruction | distractors | heldout_task |
|---|---|---|---|---|---|---|
| code | 0.00 [0.00, 0.02] (0/200) | 0.00 [0.00, 0.02] (0/200) | 0.00 [0.00, 0.02] (0/200) | 0.22 [0.17, 0.28] (44/200) | 0.00 [0.00, 0.02] (0/200) | 0.00 [0.00, 0.04] (0/100) |
| graph_single | 0.10 [0.06, 0.14] (19/200) | 0.09 [0.05, 0.13] (17/200) | 0.04 [0.02, 0.08] (9/200) | 0.11 [0.07, 0.16] (22/200) | 0.06 [0.03, 0.10] (12/200) | 0.00 [0.00, 0.04] (0/100) |
| graph_multi | 0.10 [0.06, 0.14] (19/200) | 0.09 [0.05, 0.13] (17/200) | 0.04 [0.02, 0.08] (9/200) | 0.11 [0.07, 0.16] (22/200) | 0.06 [0.03, 0.10] (12/200) | 0.00 [0.00, 0.04] (0/100) |
| graph_multi+learned_arbiter | 0.20 [0.15, 0.26] (40/200) | 0.20 [0.15, 0.26] (39/200) | 0.03 [0.01, 0.06] (6/200) | 0.20 [0.15, 0.27] (41/200) | 0.17 [0.12, 0.23] (34/200) | 0.00 [0.00, 0.04] (0/100) |
<!-- /table -->

The same runs by task:

<!-- table: eval_methods by=task methods=code,graph_single,graph_multi,graph_multi+learned_arbiter -->
| method | place | stack | sort |
|---|---|---|---|
| code | 0.09 [0.07, 0.12] (44/500) | 0.00 [0.00, 0.01] (0/500) | 0.00 [0.00, 0.04] (0/100) |
| graph_single | 0.16 [0.13, 0.19] (79/500) | 0.00 [0.00, 0.01] (0/500) | 0.00 [0.00, 0.04] (0/100) |
| graph_multi | 0.16 [0.13, 0.19] (79/500) | 0.00 [0.00, 0.01] (0/500) | 0.00 [0.00, 0.04] (0/100) |
| graph_multi+learned_arbiter | 0.32 [0.28, 0.36] (160/500) | 0.00 [0.00, 0.01] (0/500) | 0.00 [0.00, 0.04] (0/100) |
<!-- /table -->

The 4B model's first drafts are far below the reference graphs of Section 5.1, and none of
these methods completes a stack or sort episode on any split. The failures are authoring
errors, not noise. For "place the red block on the yellow square" the planner's graph looks
for a yellow cube, but the yellow square is the target zone. In the code written for the two
most common place instructions, the release step calls `close_gripper()` where it should open
the gripper, so code succeeds on place only for instructions where it wrote a correct release
(<!-- rate: eval_methods code instruction place -->0.44 [0.35, 0.54] (44/100)<!-- /rate --> on the instruction split).
Stack graphs mostly fail at the first grasp (one finger on the cube) or at its verification.

The structure critic changed some of the place graphs but no episode outcome: graph_single and
graph_multi agree on every evaluation episode. Handing grasps to the learned policy lifts the
place graphs that already reach the cube
(<!-- rate: eval_methods graph_multi nominal place -->0.19 [0.13, 0.28] (19/100)<!-- /rate --> to
<!-- rate: eval_methods graph_multi+learned_arbiter nominal place -->0.40 [0.31, 0.50] (40/100)<!-- /rate --> on
nominal) and cannot help graphs that fail before the grasp. The object split is the
exception: the planner's verify nodes accept only finger widths that fit the 4 cm cube, so
they reject the learned policy's successful grasps of the 3 cm and 5 cm variants (the
reference graphs of Section 5.1 use a band that admits every variant).

The evaluation run also scored the final graph of a deployed-condition repair loop for each
task (graph_loop in results/eval_methods.json). The loop made no accepted edits (Section 5.4),
so those graphs are simply a separate planner call per task, and their gap to graph_single is
variation between planner calls, not a loop effect. We leave them out of the tables.

### 5.3 Repair models

Fraction of held-out bugs where the model's edit raised success.

<!-- table: repair by=task -->
| method | place | stack | sort |
|---|---|---|---|
| zero_shot | 0.23 [0.12, 0.41] (7/30) | 0.40 [0.25, 0.58] (12/30) | 0.27 [0.14, 0.44] (8/30) |
| sft | 0.67 [0.49, 0.81] (20/30) | 0.63 [0.46, 0.78] (19/30) | 0.53 [0.36, 0.70] (16/30) |
| grpo | 0.73 [0.56, 0.86] (22/30) | 0.80 [0.63, 0.90] (24/30) | 0.80 [0.63, 0.90] (24/30) |
<!-- /table -->

Rows: zero_shot is the untrained 2B model, sft the SFT adapter, grpo SFT followed by GRPO.
Every problem starts from a graph that fails all rehearsal runs. The point estimates rise from
zero-shot to SFT to GRPO on every task, including sort, which no training data came from.
With 30 problems per task the intervals are wide: zero-shot vs GRPO separates on every task,
while SFT vs GRPO overlaps everywhere, so GRPO's gain over SFT is suggestive, not established.

Reading the edits (runs/*_eval_repair/problems.jsonl) qualifies this. GRPO learned to set
the broken parameter back to its default value (grasp yaw offset and depth to zero, the
verification band's upper limit near its default) on every problem of those families. That
is the right repair for these bugs and it transfers to nodes of a task never seen in
training, but the held-out bugs come from the same four families as the training bugs, so
this shows transfer across tasks and nodes, not repair of new kinds of bugs. No model fixes
the family where the band's lower limit is set too high: all three edit the upper limit. The
teacher never fixed that family either, so SFT had no examples of it. No edit deleted a
check, raised retry budgets, or padded wait times, the reward-hacking patterns we looked for.

During GRPO the four samples drawn for a prompt became identical partway through training,
which gives GRPO no learning signal on that prompt. A higher sampling temperature or more
distinct prompts per update are the obvious next steps.

### 5.4 Rehearsal repair loop

Each loop starts from a reference graph with one seeded bug (four bug families per task),
rehearses 32 instances, asks the 4B model for parameter edits at the blamed node (and the
critic for structural edits), and keeps an edit only if rehearsal success rises. Held-out
success is measured before and after on 64 other rehearsal seeds, under the loop's own
conditions: clean sensor and the ground-truth locator. Rows: the seeded-bug graph and the
loop's final graph; each cell pools the four bugs of a task.

<!-- table: loop_bugs by=task -->
| method | place | stack | sort |
|---|---|---|---|
| start_graph | 0.00 [0.00, 0.01] (0/256) | 0.00 [0.00, 0.01] (0/256) | 0.00 [0.00, 0.01] (0/256) |
| after_loop | 0.50 [0.44, 0.56] (128/256) | 0.50 [0.44, 0.56] (128/256) | 0.68 [0.62, 0.73] (174/256) |
<!-- /table -->

By bug, pooled over tasks:

<!-- table: loop_bugs -->
| method | diagonal_grasp | grasp_too_deep | tight_verify | no_release |
|---|---|---|---|---|
| start_graph | 0.00 [0.00, 0.02] (0/192) | 0.00 [0.00, 0.02] (0/192) | 0.00 [0.00, 0.02] (0/192) | 0.00 [0.00, 0.02] (0/192) |
| after_loop | 0.97 [0.93, 0.99] (186/192) | 0.97 [0.93, 0.99] (186/192) | 0.30 [0.24, 0.37] (58/192) | 0.00 [0.00, 0.02] (0/192) |
<!-- /table -->

The loop fixes the grasp-yaw and grasp-depth bugs on every task
(<!-- rate: loop_bugs after_loop diagonal_grasp -->0.97 [0.93, 0.99] (186/192)<!-- /rate --> and
<!-- rate: loop_bugs after_loop grasp_too_deep -->0.97 [0.93, 0.99] (186/192)<!-- /rate --> after the loop): the
diagnosis reports the hand-object geometry and the specialist resets the parameter behind
it. The over-tight verification band is fixed only on sort. On place and stack the
specialist edits the right parameter in the right direction, but both proposals keep the
upper limit below the cube's width, and later rounds move to other nodes. On sort its second
proposal clears the width. No loop restores the missing release. Diagnosis blames the retreat
node that follows it, where the cube leaves with the hand; the specialist raises the retreat
height, and the critic inserts motion or retry steps instead of an open-gripper node.

Run under deployment conditions instead (RGB-D locator and camera noise, starting from the
planner's graph), the loop accepted no edit on any task: held-out place success stayed at
<!-- rate: loop_deployed after_loop planner place -->0.34 [0.24, 0.47] (22/64)<!-- /rate -->. The specialist's
proposals (wider move tolerance, higher speed, perception thresholds, grasp yaw) left
rehearsal success unchanged or lowered it. Stack and sort start at zero rehearsal successes,
and no single edit moved them off zero, which leaves the loop with nothing to compare.

## 6 Limitations

- Simulation only. The camera noise model and object variants were added by us so that a
  learned executor has failures to fix; they are applied equally to all methods.
- The first learned grasp's observation is privileged (true relative pose plus small noise).
  The wrist-camera grasp detects the object itself, but its simulated depth camera has no
  minimum range (real short-range depth cameras lose depth below about 7 cm, closer than the
  top face at grasp height), and color segmentation works only because every object has its
  own saturated color.
- Object variants keep the 4 cm cube's mass and inertia.
- Only the grasp node type has a learned executor; reach and place nodes always run their
  skills.
- The oracle executor restores a simulator snapshot and is not deployable. It is a per-node
  bound, not a per-episode one: it keeps the skill's result whenever the skill passes its
  post-check, even if that grasp later slips, so an always-learned policy can beat it.
- All language models are small local Qwen models; we report no frontier-model rows. The
  authoring comparison uses one draft per instruction; all methods fail stack and sort, so
  those columns do not separate them.

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
