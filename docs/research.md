# Historical optional simulation-error protocol

This document preserves an earlier, unvalidated robosuite study proposal. It is
separate from the current Panda object-collection workbench and its policy
interface. Start with the [Panda workbench guide](demo.md) and the updated
[paper](../paper/) for the current project. This protocol is not evidence of a
research contribution or an implemented VLA training workflow.

## Original proposal: which simulation errors matter for demonstration learning?

The proposed objective was to determine whether a small amount of downstream
policy feedback can identify which simulation errors should be corrected when
collecting demonstrations under a limited interaction budget. The optional
implementation was intended as an experimental harness for this question; it has not been validated through the proposed study.

**Status:** the protocol and an initial implementation are present. No experiments
have been run for this research direction. The demonstrator, learner, and adapter
are unvalidated. There are no new research results, novelty claims, real-robot
transfer results, or VLA training results.

## Question and hypothesis

**Question:** under equal simulator interaction budgets, does selecting a
simulation correction using downstream validation success produce better
demonstration data than choosing a correction randomly or distributing data
collection uniformly across corrections?

**Hypothesis to investigate:** the best correction depends on the task and policy;
correcting a consequential error can be more useful than collecting additional
data under an irrelevant correction. The cost of identifying that error can also
outweigh the benefit. The protocol measures both possibilities.

The current implementation is a single-task sim-to-sim pilot. The target is the
nominal robosuite Panda Lift environment. Its parameters are not measurements of
a physical Panda. Artificially introduced errors make causal interventions
possible, but do not establish that these are the dominant errors in practice.

## Implemented experimental conditions

The optional backend uses **robosuite 1.5.1, MuJoCo, Panda, and Lift**, separately
from the illustrative RG Mobile Duo Studio. It loads the upstream robot and task
assets; no new robot CAD or fabricated hardware is claimed.

Four interventions are applied after reset and before the initial image:

- `contact_friction_scale`: multiply the sliding-friction coefficient of every
  geom. Torsional and rolling coefficients, geometry, and solver settings stay
  fixed. This is a contact-parameter mismatch, not a higher-fidelity solver.
- `camera_x_offset_m`: displace the fixed `agentview` camera along world x. This
  changes the rendered viewpoint; it does not change policy input calibration
  metadata, because this baseline does not consume calibration.
- `light_scale`: scale scene-light diffuse and ambient intensities. Materials,
  textures, shadows, and image resolution are fixed.
- `action_delay_steps`: insert a FIFO queue before the OSC command controller.
  Its initial entries are zero commands. Delay is measured in control steps,
  not physics substeps. At 20 Hz, two steps correspond to 100 ms.

The reference values are `(1, 0 m, 1, 0 steps)`. The example biased condition is
`(0.4, 0.04 m, 0.5, 2 steps)`. These are declared experimental choices, not
calibrated reality gaps. A refinement replaces the selected parameter with its
reference value. All other biased parameters remain in effect.

To isolate one error, set the other three parameters to their reference values
in a separate configuration. Only non-reference parameters become candidate
refinement axes. At least one active error is required. Use distinct study names
for independently declared studies.

The comparison strategies are:

1. **Biased:** collect all demonstrations in the biased condition.
2. **Reference:** collect all demonstrations in the nominal target condition.
   This is a diagnostic comparator, not a theoretical upper bound. Knowledge of
   the correct target parameters is provided by construction, with no calibration
   cost charged; this comparator is not a practical calibration method.
3. **Uniform:** cycle through the four single-axis corrections, collecting an
   equal number of episodes when the budget is divisible by four.
4. **Random:** choose one single-axis correction per training seed, then use it
   for the full collection budget.
5. **Guided:** collect pilot data separately for the biased condition and each
   single-axis correction. Train one policy per candidate, evaluate all candidates
   on paired nominal validation episodes, and spend the remaining budget on the
   highest-scoring candidate. Ties prefer biased, then the declared axis order.
   Train the final policy on **all** pilot and subsequent collection episodes.

Guided is a deliberately simple one-round selection baseline. It does not infer
physical parameters, diagnose individual failure causes, explore combinations,
or implement a novel adaptive-fidelity algorithm. Subsequent research should
compare a proposed method against it.

## Budget, pairing, and outcomes

Every episode executes exactly the configured horizon, including after an early
lift, so every budgeted rollout consumes the same number of control steps. An
unexpected early termination aborts the run. Demonstration collection and guided
selection both count against the same per-strategy budget. Failed demonstrations
are retained without retries or success filtering.

For the example, every strategy spends **48 rollouts / 9,600 control steps** per
training seed. Guided spends 15 rollouts collecting candidate data and 15 on
selection, leaving 18 additional demonstrations: 33 training episodes total.
Other strategies collect 48 training episodes. This accounts for guided's
interaction overhead rather than giving it free validation data.

Policy training compute is **not** equalized. Training fits and elapsed training
and simulation time are reported separately. These experiments cannot establish
an improvement under equal GPU-hours or total monetary cost. Parameter correction
also has no modeled engineering or measurement cost. A future cost-allocation
method must measure and charge those costs explicitly.

Training, selection, and held-out initial conditions use separate deterministic
seed namespaces. Candidates share pilot initial conditions. All final policies
share the same held-out cases across training seeds and strategies. The runner
checks hashes of initial joint positions and velocities when an episode seed is
reused. This checks state pairing, not bitwise reproducibility across machines.

All final policies are trained and every selection is locked before any held-out
evaluation begins. Do not tune the configuration on these held-out scores; use
a new held-out namespace for a subsequent confirmatory study.

The primary outcome is **sustained final success**: robosuite's sparse lift success
must be true at every step of the last hold window. The example uses 10 steps
(0.5 seconds). The report also records any earlier lift, sparse return,
demonstrator success, and descriptive failure categories (`never_lifted` and
`lift_not_held`). These categories do not diagnose collision or grasp mechanics.

Summaries include per-training-seed scores and paired guided-minus-baseline
differences. Exploratory 95% percentile intervals use a crossed bootstrap over
training seeds and shared held-out cases. No interval is produced for a single
training seed. Three seeds, twenty held-out cases, multiple comparisons, and one
task are insufficient to establish broad improvements.

## Learner and recorded data

The built-in learner is **visual nearest-neighbor behavior cloning**, not a VLA:
8x8 RGB block means, robot proprioception, and previous requested action select
the closest recorded action. The image and proprioception/action feature groups
are normalized separately. The instruction is supplied through the interface but
this baseline ignores it; one fixed instruction cannot test language grounding.

The state-based heuristic demonstrator has privileged access to cube position.
The learned policy receives only RGB, proprioception, previous requested action,
and instruction. Object coordinates, rewards, and success are withheld from
`policy.act`. External trainers must preserve that boundary.

Each training `.npz` contains `H+1` RGB/proprio/previous-action observations and
`H` requested actions, executed actions, rewards, and success flags. Transition
`t` is `(observation[t], requested_action[t], observation[t+1])`; the executed
command can differ because of delay. Neither command is a measured displacement.
The archive also records initial joint state, instruction, seed, and profile.
These research archives have a different robot and controller from Studio ZIP
episodes and must not be passed to its RG Mobile Duo dataset validator.

## Using the implementation

Create a separate Python environment and install the project and optional
backend dependencies. This does not change the standard Studio installation:

```sh
python3.12 -m venv .venv-research
.venv-research/bin/pip install -e . -r examples/research/requirements.txt
```

Offscreen MuJoCo rendering requires an available OpenGL backend. Follow the
[upstream installation instructions](https://robosuite.ai/docs/installation.html)
for your OS. Configure the graphics backend before launching Python. A GPU is
not universally required, but rendering support is required for this RGB study.

Inspect the budget without loading robosuite or starting simulation:

```sh
.venv-research/bin/python -m robogogogo.research plan \
  --config examples/research/panda_lift.json
```

Only the explicit `run` command starts collection, training, and evaluation:

```sh
.venv-research/bin/python -m robogogogo.research run \
  --config examples/research/panda_lift.json \
  --output runs/research/panda-lift-v1
```

The example schedules 1,020 rollouts, or 204,000 control steps, across three seeds
and five strategies. It is a study configuration, not a quick smoke test. Output
must be a new directory; interrupted or completed studies are never overwritten
or silently resumed. A failed run has no completed `results.json`.

Outputs include the declared plan, backend/controller and package versions,
training episodes and checkpoints, selection decisions, per-episode outcomes,
and aggregate JSON summaries. They belong under the Git-ignored `runs/` directory.
No generated result files or evidence directory are added to the public source.

Recompute the summary of a completed study:

```sh
.venv-research/bin/python -m robogogogo.research summarize \
  runs/research/panda-lift-v1/results.json
```

### Connecting a stronger policy

Set `trainer` and `loader` in a study config to local `module:function` callables.
The runner calls:

```python
policy = train(episode_paths, output_dir=checkpoint_directory, seed=training_seed)
policy.reset()  # once per rollout
action = policy.act(observation, instruction)
# After every strategy has been fitted, reload one final policy at a time:
policy = load(checkpoint_directory)
```

`observation` has `rgb` (uint8 HWC), `proprio` (23 values: Panda joint-position
cosines and sines, end-effector site position/quaternion, and gripper positions),
and `previous_action`
(seven normalized commands). Preserve this ordering from `backend.json`.
`act` returns seven finite commands within robosuite's action bounds. Invalid
commands abort the run instead of being silently clipped. A trainer must save its
own checkpoint, honor the supplied seed, and report its configuration in its
checkpoint directory. The loader restores that policy for held-out evaluation;
only one final policy is resident at a time. An optional `policy.close()` method
can release external resources. Do not load untrusted trainer modules or
checkpoints.

A diffusion policy or pretrained VLA needs a real adapter for its action
representation, image preprocessing, instruction handling, and checkpoint format.
The import hook alone does not implement those integrations.

## What would make this a research contribution?

Before a main study, establish that the expert reliably lifts in the nominal
condition and that the chosen learner has meaningful, non-saturated performance.
If it fails everywhere, improve the demonstrator/learner before comparing
simulation conditions. The nearest-neighbor baseline is an implementation aid,
not evidence that modern robot policies need a particular fidelity level.

For a substantive study, extend the controlled design to multiple contact and
placement tasks, multiple policy families, error magnitudes, and interacting
errors. Include one-factor ablations to distinguish which errors cause an effect.
Predeclare an effect size and budget before confirmatory runs. A useful method
must beat the simple guided baseline as well as random/uniform strategies after
charging selection and training costs. Publish negative findings when supported.

Claims about real-world usefulness require measured/calibrated robot conditions
and held-out physical trials. CAD supplies geometry; it does not determine
friction, actuator latency, or camera calibration. Claims about VLA generality
require actual VLA training/evaluation and tasks with varying instructions.

World-model data could later be an additional source condition, with the same
downstream policy evaluation and explicit generation/adaptation costs. There is
no world-model integration in this implementation.

## Closest prior work and remaining novelty assessment

- [SIMPLER](https://arxiv.org/abs/2405.05941) studies simulated evaluation of
  real-world manipulation policies and visual/control disparities. This project
  instead proposes studying collection choices and downstream learning under a
  charged selection budget; that distinction alone does not establish novelty.
- [VISER](https://arxiv.org/abs/2605.06311) analyzes lighting/material effects and
  visually realistic VLA evaluation. Showing that appearance matters would not
  be a new contribution.
- [How Should a Simulation-to-Reality Transfer Budget Be Spent?](https://arxiv.org/abs/2606.22062)
  studies identification versus randomization budgets in a controlled pendulum
  setting. Budget allocation itself is not a new question.
- [WorldEval](https://arxiv.org/abs/2505.19017) uses world models for robot-policy
  evaluation; [VLAW](https://arxiv.org/abs/2602.12063) improves policies using
  learned-simulator rollouts. Merely adding a world model is not a contribution.

A closer review of task-driven simulator adaptation, active system identification,
and data selection is needed before claiming a distinct method. This initial
implementation defines an experiment that can reject an idea, rather than assume
the project's existing software constitutes a scientific result.

Backend API references: [robosuite environments](https://robosuite.ai/docs/modules/environments.html),
[controllers](https://robosuite.ai/docs/modules/controllers.html), and
[versioned Lift source](https://github.com/ARISE-Initiative/robosuite/blob/v1.5.1/robosuite/environments/manipulation/lift.py).
