# robogogogo

**A robotics control and perception platform, built on native Python/JAX physics.**

A robot is a system: sensing, perception, task planning, motion control, actuation,
and evaluation. VLA is one possible policy within that system. robogogogo supports
classical rule-based controllers and model adapters through shared contracts.
No external physics engine is used. Optional OpenCV powers stereo perception only.

**Robotics Studio** is a Next.js/TypeScript workbench connected to Python/FastAPI/JAX.
The current implementation supports multi-arm reaching, calibrated camera rigs,
stereo estimation, renderer-derived depth, supervised joint actions and aligned
recordings. This is an early research foundation: full humanoid locomotion,
general mesh/contact planning, pretrained VLA adapters, training jobs and RL trainers
are not implemented yet.

## Run Robotics Studio

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[studio,dev,perception]'
cd studio
npm ci
npm run build
cd ..
python -m robogogogo.server
```

Open **http://127.0.0.1:8765**. The backend compiles the JAX robot model on startup,
then serves the Next.js static export and API together. Fonts and JavaScript are
bundled locally; no CDN or external model service is needed. Node.js is needed to
build the UI, not to run the packaged static interface.

The installed command is also `robogogogo-studio`. The default storage directory is
`runs/studio/`, outside version control. Use `--projects PATH` or `--port 8765` to
configure the local server. One server is one shared workspace, including across
browser tabs. It binds to loopback, not a public interface.

## First robot episode

1. In **Episode**, choose a reach-task fixture and enter its language instruction.
2. Click **Start new episode**. The robot resets, and the fixture sets a reachable target.
3. Click **Capture observation** to record RGB, instruction, joint positions, and velocities.
4. Enter a joint-delta action and click **Execute + capture next RGB**. Python advances
   the controlled dynamics, and Studio records the next camera frame at the resulting state.
5. Alternatively, use **Run rule-based reach** to exercise the entire loop with
   a scripted IK baseline. This is explicitly **not a learned VLA policy**.
6. Click **Save & export episode**. The ZIP is saved locally and downloaded; it includes
   PNG frames, a versioned JSON manifest, aligned transitions, and physics parameters.
7. Open **Data** in the workspace sidebar to download saved episodes again.

The language instruction is a policy input. The application does not parse it into
a motion command. The named fixture supplies the task target. Success means every tool
position within 2 cm and joint-velocity norm below 0.1 in this reaching fixture.
This does not establish grasping or manipulation capability.

The robot inspector and joint controls are debugging tools. Changing physical
state through those controls ends the current VLA episode rather than quietly
mislabeling unrecorded actions. Start a fresh episode afterward.

## One arm or multiple arms

Select **Workspace → Robot embodiment** to switch between the single six-joint arm
and a dual-arm rig with twelve joints. Switching embodiments starts a new episode;
export the previous episode first. **Robot → Control arm** selects the arm for
joint goals and Cartesian target editing. IK considers all tool targets together;
inspect residuals because conflicting or unreachable goals may not converge.

The same model supports larger fixed-base articulated trees, subject to Studio's
48-moving-joint / 128-link resource limit. These limits are not a performance or
physical-fidelity guarantee. Import a multi-arm URDF through `/api/robot`, optionally
providing explicit tool definitions:

```python
payload = {
    "xml": urdf_text,
    "arms": [
        {"name": "left_arm", "tool": "left_tool"},
        {"name": "right_arm", "tool": "right_tool"},
        {"name": "auxiliary_arm", "tool": "auxiliary_tool"},
    ],
}
```

Without definitions, each moving terminal branch becomes an arm group. For
humanoids with head, leg, or finger branches, **supply explicit arm tools**. Group
joint membership follows the tool's ancestry; a torso joint can belong to multiple
groups. `models.multi_arm_urdf(count)` creates reference rigs with one to four arms.

VLA actions remain one vector in `action_spec.names` order. `action_spec.arms` maps
named arms to `joint_indices`; shared joints occur only once in the vector.
Every action advances all arms together. The reaching fixture gives each tool its
own target and succeeds only when **all** tool errors are below 2 cm and the joint
velocity norm is below 0.1. Reward is the negative sum of squared tool errors.

Snapshots and datasets include `targets`, `tool_positions`, and arm definitions;
transitions include per-arm `target_errors`. Checkpoints restore all targets and
groups. Physics NPZ exports include `arm_targets` with shape `[samples, arms, 3]`;
`target` remains the first arm's target for legacy consumers.

This supports multi-arm joint dynamics, reaching and frictional grasp handover.
Finger/object contacts are simulated; general arm-to-arm contact forces are not solved. Coordinated tasks
add capsule-based collision avoidance and combined-joint motion planning.
Multiple arms still share one
fixed-base robot model, not independently floating robot agents.

## Multiple cameras and humanoid sensing

Studio supports **1–8 named RGB cameras per episode**. Built-in arm rigs now
start with a **head stereo pair plus one wrist camera per arm**. Their `head` link
is a visible fixed camera mast for the demo, not an articulated humanoid neck.
Imported humanoids can use their actual head and hand links.

Under Episode, choose **Mounting preset**, then expand each camera to edit its
parent link, XYZ mounting offset, roll/pitch/yaw, and vertical field of view.
Click **Start new episode** to apply changes. Camera outlines in the viewport
show active installation locations and viewing directions; these guides are
excluded from captured RGB. Camera orientations never track task targets.

The illustrative head baseline is **64 mm**, pitched **40° downward**. Wrist
cameras sit **80 mm above their tool frame**, pitched **20° downward**. These are
example bracket dimensions, not calibrated hardware settings. Replace them with
measured extrinsics and your lens's field of view before matching a real robot.
All camera images use the same paused simulation state and are submitted in one
bundle. Missing, extra, or invalid views reject the entire observation.

Mount rotations use `Rz(yaw) Ry(pitch) Rx(roll)` in radians: +X forward, +Y left,
+Z up, positive pitch downward. The optical frame remains +X right, +Y down,
+Z forward. `rpy` is the orientation source of truth when provided; the API
resolves legacy `look_at`/`up` fields from it for rendering compatibility.

Custom rigs use `POST /api/vla/reset` with `cameras`, for example:

```json
{
  "instruction": "Reach the target using all camera views.",
  "cameras": [
    {"name": "head_left", "mount": "head", "position": [0.08, 0.032, 0], "rpy": [0, 0.698132, 0]},
    {"name": "head_right", "mount": "head", "position": [0.08, -0.032, 0], "rpy": [0, 0.698132, 0]},
    {"name": "left_wrist", "mount": "left_hand", "position": [0, 0, 0.08], "rpy": [0, 0.349066, 0]}
  ]
}
```

Mount names must exist in the robot; the bundled fixtures have `head` and tool
links, but do not have `left_hand`. Omit `mount` for a world-fixed camera. Position and rotation are expressed in the mount's coordinates; legacy look-at
and up coordinates are still accepted. Width/height default to
320×240 (supported range 32–640 by 32–480); FOV and clipping planes are configurable.
The example assumes those links use +X forward and +Z up; adapt it to your robot.

Capture uses `images: {"head_left": "data:image/png;base64,...", ...}` plus one
snapshot `sequence`. Custom-rig archives use `robogogogo-episode-v2`, with named
frame paths, pinhole intrinsics, and per-observation camera-to-world extrinsics.
Optical coordinates are +X right, +Y down, +Z forward. No lens distortion, rolling
shutter or hardware clock synchronization is simulated. Browser rendering
is sequential, but simulation time and robot state are identical across the bundle.

The Python reader exposes `observation.images[name]`, `observation.cameras`, and
`observation.camera_poses`. It also reads legacy v1 archives; `image_png` remains
the first camera for compatibility. Training adapters should use `images` so they
do not silently discard other views. Camera configurations persist in checkpoints.

This is sensing infrastructure for future humanoid work. The current physics
still uses fixed-base articulated robots; floating-base dynamics, walking,
balance and articulated hands remain unimplemented. The handover fixture uses
force-limited parallel-jaw grippers and one movable object.

## Observation and action contract

Each observation contains:

- Named camera views: PNG frames in sRGB, independent of viewport orbit; legacy single-view episodes use `agent_view`.
- `instruction`: the episode's UTF-8 text instruction.
- `joint_position`, `joint_velocity`: explicit joint-coordinate proprioception.
- Simulation timestamp, tool position, and task target.

The initial action representation is **`joint_position_delta`**, in radians for
revolute joints and meters for prismatic joints, with no normalization. Actions
are relative to measured joint positions. The default duration is 10 control
ticks: 100 ms with the default 2 ms physics step and five substeps/tick. Deltas are
bounded to 0.12 rad or 0.02 m, then clipped to joint limits. Both requested and
applied values are recorded. A torque controller executes the resulting goal.

Gripper commands currently run as guarded handover task-stage events; there is no free-form policy gripper action. Do not directly send a pretrained model's
Cartesian/gripper action vector to this interface. Model adapters must explicitly
handle robot embodiment, action representation, scaling, normalization, camera
preprocessing, and action-chunk timing.

A browser camera frame is accepted only for the matching paused backend snapshot.
Actions require the latest observation at the current physical state. Each stored
transition references its before/after camera observations. Incomplete exports are
marked in the manifest; the dataset reader rejects unaligned transitions.

## Use collected data in Python

```python
from robogogogo.datasets import iter_transitions

for transition in iter_transitions('episode.zip'):
    instruction = transition.observation.instruction
    rgb_png = transition.observation.image_png
    q = transition.observation.joint_position
    action = transition.action
    next_rgb_png = transition.next_observation.image_png
    # Your training, evaluation, or offline-RL adapter consumes these values.
```

```bash
python examples/read_vla_episode.py path/to/episode.zip
```

Images remain PNG bytes so downstream packages choose their own decoding and
preprocessing. No trainer, normalization, or model-specific action convention is
silently imposed. Source labels distinguish `manual`, `scripted_reference`, and
`policy` actions. Only use `policy` for actions produced by an actual model.

`robogogogo.vla.Policy` defines the Python adapter interface:

```python
from robogogogo.vla import Observation, ActionChunk

class MyPolicy:
    def predict(self, observation: Observation) -> ActionChunk:
        # Invoke your model and explicitly convert its actions to this robot's contract.
        raise NotImplementedError
```

This is the integration boundary, not a bundled model implementation.

## Local API

Interactive API documentation is available at **http://127.0.0.1:8765/docs**.

- `GET /api/snapshot`, `WS /api/ws`: robot, task, episode, and live state.
- `POST /api/vla/reset`: instruction and task fixture.
- `POST /api/vla/observe`: PNG camera frame plus matching snapshot sequence.
- `POST /api/vla/act`: bounded joint delta, observation index, duration, and source.
- `GET /api/vla/reference-action`: explicitly labeled scripted IK action.
- `POST /api/vla/save`, `GET /api/vla/episodes`: persist and list episode datasets.
- `GET /api/vla/episodes/{id}`: download an episode archive.
- `/api/projects`: save and restore workspace state checkpoints.
- `/api/robot`: import the supported fixed-base URDF subset.
- `/api/export`: download the underlying physics trajectory as NPZ.

Checkpoint saves and episode exports have different purposes: a checkpoint
restores robot state and goals; a dataset export preserves the recorded visual
transitions. Neither substitutes for the other.

The browser keeps 600 recent state frames for inspection. VLA episodes have a
250-action horizon and a 501-observation cap. Physics recording has an 18,000-tick
cap. Long-running collection needs a streaming dataset pipeline in a later release.

## Develop the Next.js interface

Run the Python backend in one terminal. In another:

```bash
cd studio
npm run dev
```

Open **http://127.0.0.1:3000**. Next.js proxies HTTP API calls to the Python server;
the browser opens its WebSocket directly to port 8765. Optional development
configuration: `ROBOGOGOGO_API_URL` for the HTTP proxy target and
`NEXT_PUBLIC_ENGINE_WS` for a custom WebSocket URL. The local API permits the
standard localhost/127.0.0.1 development origin on port 3000.

`npm run build` uses Next.js App Router static export and copies it into the Python
package's generated `web/` directory. The 3D interface is a client component;
physics does not run in Next.js route handlers. The source, lockfile, and package
configuration are versioned; generated frontend assets are ignored by Git.
Build the interface **before** building a distributable Python wheel.

## Validation

```bash
ruff check src tests examples
ruff format --check src tests examples
MPLCONFIGDIR=/tmp/robogogogo-matplotlib pytest -q
cd studio
npm run format:check
npm run typecheck
npm run build
```

Tests cover the underlying dynamics and gradients, API state transitions,
WebSocket clock behavior, checkpoint persistence, URDF validation, recording
limits, camera/action alignment, stale-observation rejection, action bounds,
reference collection, episode archives, and dataset reading. Browser checks cover
the real WebGL camera and user workflows; API tests use valid synthetic PNGs for
contract verification and do not claim visual-model performance.

## Direction and current limits

The package is organized around robot control and perception:

- **Environment runtime:** `spatial`, `control`, `models`, `session`, `env3d`.
- **Model interface:** `vla.Observation`, `ActionChunk`, and `Policy`.
- **Data:** episode recorder/exporter in `vla`, aligned readers in `datasets`.
- **Studio:** `studio/` (Next.js/TypeScript) and `robogogogo.studio` (FastAPI).

### Code boundaries

- `simulation/`: model definitions, kinematics, contacts, dynamics, and integration.
  The `spatial` module preserves the public imports used by existing scripts.
- `session.py`: simulation state and trajectory recording, independent of HTTP and UI.
- `vla/contracts.py`: model-facing observation/action types and policy protocol.
- `vla/recording.py`: episode alignment and archive serialization.
- `studio/schemas.py`: validated request and checkpoint schemas.
- `studio/workspace.py`: the single local workspace and simulation lifecycle.
- `studio/episodes.py`: VLA collection operations under the workspace lock.
- `studio/api.py`: HTTP routes, WebSocket streaming, and application lifespan.
- `server.py`: CLI entry point and compatibility exports.
- `studio/src/state/`: browser workspace state and orchestration.
- `studio/src/components/`: focused workbench panels.
- `studio/src/hooks/`: live connection lifecycle; `api.ts` owns HTTP transport.

Python modules use Ruff; the interface uses Prettier and strict TypeScript,
including unused-code checks. Run `ruff format src tests examples` and
`npm --prefix studio run format` before submitting changes. Keep numerical
kernels independent of web dependencies and route new episode operations through
the service layer so observation/action ordering remains serialized.

Next development should add model-specific inference adapters, streaming datasets
and data QA, training/fine-tuning integrations, batched VLA evaluation, and online
RL. These should share the same task/observation/action contracts, with explicit
versioning when embodiments or action spaces change.

Current environment coverage includes reaching, collision-aware inspection and a dual-arm
pick–handover–place fixture with compliant finger contact and friction. There is no general mesh contact solver,
headless RGB renderer, pretrained VLA inference service, training pipeline, or
online VLA RL trainer. The browser camera is required for today's visual episode
collector. JAX supplies compiled numerical kernels; the physics algorithms are
implemented in this package's Python source.

See [physics runtime details](docs/physics.md) for equations, model construction,
legacy demos, and numerical limitations.

## Composable control system

`robogogogo.contracts` exposes `Observation`, `ActionChunk`, and `Policy` for both
classical controllers and learned adapters (legacy `robogogogo.vla` imports remain
supported). `ControlSystem` selects an explicitly named policy, validates its
proposal, and bounds each joint increment and accumulated joint position.
`stop()` rejects further proposals until `resume()`. This software supervisor is
not a physical emergency stop or hardware safety controller. An optional
`collision_world` adds geometric checks to each proposed action.

`ReachPolicy(model, arms)` uses simultaneous position IK for one or multiple arms.
It reads privileged target positions. `JointWaypointPolicy` is a deterministic
state machine that waits for position and velocity tolerances before advancing.
It holds at completion and has an explicit `reset()` for a new episode. Neither
baseline parses language or performs visual manipulation. Learned adapters can
implement the same `predict(observation)` method; no trained model is bundled.

The implemented pipeline is observation → selected policy → action supervisor →
computed-torque joint controller → native dynamics. Coordinated inspection now
adds task stages and collision-aware motion planning. General state estimation,
semantic task planning and hardware actuation remain future integrations.

```python
from robogogogo.policies import JointWaypointPolicy
from robogogogo.system import ControlSystem

policy = JointWaypointPolicy([[0.2, -0.1], [0.4, 0.0]], max_delta=[0.1, 0.1])
system = ControlSystem({"waypoints": policy}, lower=[-1, -1], upper=[1, 1],
                       max_delta=[0.1, 0.1])
# decision = system.step(observation, controller="waypoints")
# Send decision.action through the simulator's aligned action interface.
```

## Stereo depth and evaluation

Install `pip install -e '.[perception]'`. In the episode panel, select the head
left/right cameras and click **Capture & evaluate stereo**. Both RGB frames and
renderer ground-truth depth use the same paused simulation state. The panel shows
estimated depth, ground truth, disparity, valid coverage, MAE and RMSE. Low-texture
surfaces legitimately yield sparse or empty estimates; unknown pixels stay masked.

`robogogogo.perception.estimate_stereo` rectifies calibrated pinhole RGB images,
runs OpenCV StereoSGBM, applies left/right consistency and texture checks, and
returns disparity, optical Z depth in meters and a validity mask. The supported
pair is horizontal and ordered left/right; distorted real images need calibrated
undistortion upstream. Output coordinates are the rectified left camera. Ground
truth is rotated/remapped into that same frame before comparison, and never enters
the stereo matcher. The renderer's cosmetic meshes define this visual ground
truth; it is not a collision-distance measurement.

RGB captures also save little-endian float32 ground truth (NaN for no surface).
Exports add `depth/{camera}/{observation:06d}.npy` and `depth_contract` metadata to
the existing v1/v2 archives. Older readers can ignore this optional addition.
`iter_transitions(path)` excludes privileged depth by default; use
`iter_transitions(path, include_ground_truth=True)` for evaluation. Estimated maps
are available from the Python perception API; Studio previews are not exported.

Neutral `/api/episodes/{reset,observe,act,save}` routes coexist with `/api/vla/*`.
`GET /api/control/rule-based-action` supplies supervised reach proposals;
`POST /api/perception/stereo` evaluates a captured observation with `left`, `right`
and `observation`. Actions from the Studio rule controller are labeled
`rule_based`, separate from learned `policy`, `manual`, and legacy
`scripted_reference` data.

## Coordinated dual-arm tasks and collision avoidance

Load the **Dual arm** embodiment, then use **Plan coordinated task** in the episode
panel. The planner constructs the full task before replacing the existing episode;
save any current recording you want to keep first. **Run coordinated task** records
synchronized cameras while the two arms perform a position-only shared-station
inspection fixture:

1. Both arms move to separate staging positions.
2. The left arm visits and dwells at the shared station; the right arm waits clear.
3. Both arms clear the station.
4. The right arm visits and dwells; the left arm waits clear.
5. Both arms return to their starting tool positions.

Stage completion requires joint position, velocity, tool position and dwell-time
conditions. Station ownership, stage completions, path method and clearance are
visible in Studio and exported in episode metadata. Pause acts between recorded
control actions; resume continues the current stage. Blocked tasks require
replanning. Ordinary manual/debug actions cannot override an active task: start a
new episode to release task ownership. A physics reset or embodiment change clears
it. The inspection recipe is coordinated positioning; the separate handover recipe adds
frictional grasping and object transfer, described below.

`robogogogo.planning` provides:

- `Capsule`, `SphereObstacle`, `CollisionWorld`: primitive collision geometry,
  robot/self/obstacle/floor clearance, and joint-limit validation.
- `plan_motion`: combined-joint-space direct planning with seeded bidirectional
  RRT-Connect when the direct edge is blocked. Every returned edge, including
  shortcuts, uses adaptive subdivision and a conservative displacement bound.
  Uncertified edges are rejected. Search is bounded by iteration and wall-time
  budgets; failure does not prove a path is impossible.
- `TaskPlan`, `TaskStage`, `TaskExecutor`: feedback-driven multi-stage execution,
  explicit shared-station ownership, dwell conditions, pause and timeout states.
- `plan_shared_inspection`: the two-independent-arm fixture used by Studio.

The Studio fixture encloses primitive robot visuals with capsule proxies and adds
an orange sphere obstacle below the station. Required separation is **15 mm**.
Attached links within two tree edges are explicitly excluded because the demo's
joint housings overlap; the exclusions and full proxy geometry are exported.
These are planning approximations, not a replacement for calibrated URDF collision
meshes. Decorative frontend shells, cameras, cables, arbitrary meshes and carried
objects are not covered. Supply measured proxies and justified exclusions for
other robots. The lower-level API accepts explicit capsules and sphere obstacles;
the Studio task currently uses the fixed inspection fixture. Standalone reach and
manual controls do not automatically enable this collision world.

`ControlSystem(..., collision_world=world)` checks bounded policy proposals.
`Session.advance(collision_world=world)` additionally predicts the native dynamics
at every physics substep, checking states and connecting edges before committing
that control tick. If the guard rejects motion, the simulator freezes at its last
accepted state; this is **not a physical braking controller or hardware safety
certification**. Clearance diagnostics report sampled simulated states, while
edge checks enforce the configured margin between them. No body-body contact
forces are added by the planner.

A browser-free demo needs only the core Python/JAX package:

```bash
python examples/coordinated_inspection.py --output runs/coordinated-inspection.npz
```

Studio APIs: `POST /api/tasks/plan` accepts an optional world-frame `station`
(default `[0.55, 0, 0.75]`) and deterministic `seed`; `POST /api/tasks/step` requires
the latest aligned `observation` index. `/api/tasks/pause` and `/api/tasks/resume`
control execution. Snapshots include `coordination`; exported task actions use
source `task_planner`, per-transition stage/owner/clearance metadata, the full
`task_plan` and final `task_progress`. The existing dataset reader remains compatible.

## Grasping and object handover

In Studio, choose the dual-arm embodiment, start a fresh episode, select
**Pick → hand over → place**, then plan and run. Set the force per jaw and static/
sliding friction coefficients in **Finger contact** before planning. The 16 stages
include collision-checked pregrasp poses, straight-in approaches, level lift and
lowering, and straight-out withdrawals. Actions use 20 control ticks (200 ms).

The 250 g capsule always has independent six-degree-of-freedom dynamics. Six
samples on each finite finger pad generate unilateral spring/damper normal
forces. Tangential spring history models stiction; a Coulomb return map limits
friction during sliding. Contact separation clears the friction history. Jaw
actuators have finite effective mass, force limits and a rate-limited position
command. There is **no rigid attachment** and no merged payload inertia.

Object and arm contact reactions are integrated together with 0.2 ms contact
substeps. The joint controller uses measured contact-load feedforward, subject
to motor torque limits. Simultaneous donor/receiver contacts share the load
through these dynamics. The receiver must sustain bilateral contact before the
giver opens; failed receiver acquisition leaves the giver closed. Release and
handover wait for the outgoing jaw to open and lose contact. Contact loss and
slip are recorded; neither ownership labels nor task success flags exert forces.

The collision guard checks the **actual object pose**, including slip, against
robot proxies and support boxes. Intentional finger/object contact and the
specified payload/support contact are allowed, with up to 3 mm support-contact
penetration. The displayed effective clearance includes that allowance.

Episodes export contact normal/friction forces, relative speed, jaw positions,
friction and actuator settings, object pose/velocity, arm reaction torques and
handover events alongside camera observations. Project checkpoints still restore
robot state only; export the episode to retain the manipulation record.

This remains a **sampled compliant-contact approximation**, not experimentally
calibrated hardware physics. It supports a capsule and flat inner finger pads,
box supports and a ground plane; arbitrary meshes, outer finger/palm impacts,
soft-pad deformation fields are not implemented. Support rolling resistance is
a configurable kernel coefficient, not a measured material model.
The arm Jacobians and effective inertia are linearized over each 2 ms physics
step. Study timestep and stiffness sensitivity before using it for research claims.
Rigid-body energy telemetry excludes energy stored in contact springs.

Try the browser-free fixed-hand benchmark (native Python/JAX only):

```bash
.venv/bin/python examples/friction_grasp.py --friction 0.8 --force 40
.venv/bin/python examples/friction_grasp.py --friction 0 --force 40 --output runs/frictionless.npz
```

The first holds the bar; the second lets it slip out and fall. Each writes an NPZ
trajectory with jaw widths, object positions and contact measurements.

Run contact regressions with:

```bash
.venv/bin/pytest tests/test_grasp_contact.py tests/test_coordinated_task.py -q
```

Tests cover holding versus frictionless falling, weak grips, Coulomb limits,
separation, timestep refinement, arm reaction forces, failed receiver acquisition
and the recorded full handover. See [contact model details](docs/physics.md#finger-contact-and-friction).

Adaptive grasping is enabled for Studio handovers. Under **Finger contact**, set
an initial jaw force, actuator limit, and calibrated object force threshold.
Feedback pauses arm motion during slip recovery and records measured peak loads;
unrecoverable slip or overload blocks the task. Force thresholds are configurable
safety envelopes, not a fracture or material-damage model.

Compare adaptive and fixed-force grasps after a friction disturbance:

```bash
python examples/adaptive_grasp.py
```

Studio now includes **soft foam, soft elastomer, and fragile shell** material
presets under **Material & finger contact**. The Python/JAX contact solver tracks
local viscoelastic compression, permanent dents, pressure damage, and stiffness
loss. Adaptive grasping unloads on material risk; detected damage blocks the task.
The rendered object and exported camera/depth observations show deformation, with
red regions marking modeled damage.

```bash
python examples/material_grasp.py --material soft_foam
python examples/material_grasp.py --material fragile_shell --force 40 --no-adaptive
```

These are illustrative local-indentation models, not calibrated material data or
volumetric FEM. They do not yet simulate bending, tearing, or fracture fragments.
See [material mechanics and validation limits](docs/physics.md#deformable-material-surfaces-and-damage).

For a visual material test, load **Dual arms**, choose a material, and run
**Material grasp → lift** (selected automatically). This verifies lifting without
modeled damage. The full handover remains a separate, guarded experiment and may
stop when the soft object's grip cannot support the transfer motion.

### Structural Lab (experimental)

Open **Structural Lab** from Studio to run a contact-driven cantilever lift,
two-tool tear or destructive crush. Deformable cells bend, cohesive bonds fail
irreversibly, and disconnected pieces retain their mass and move independently.
Choose a material and experiment, **Reset specimen**, then **Run**. Export the
trajectory as JSON. For a headless destructive test:

```bash
.venv/bin/python examples/structural_grasp.py --material fragile_shell --recipe crush --output /tmp/fracture.json
```

The bench is also available in articulated handover: choose a deformable material
and enable **Structural dynamics (experimental)** in the coordination panel.
Both arms use structural contact forces, damage feedback and guarded load transfer.
It uses predefined fracture interfaces and illustrative material parameters;
see `docs/physics.md` for model assumptions and current limits.


Run the native two-arm structural lift/handover regression:

```bash
.venv/bin/python examples/structural_handover.py --material soft_elastomer --force 16 --lift 0.1
```

The donor gradually unloads and verifies receiver support before opening. Rising
structural risk holds both arms and reduces force. Structural states and adaptive
events are included in episode snapshots; the main view and cameras share the
same deforming object geometry. This remains an experimental, uncalibrated model.

### Coordinated grasp recovery

Enable **Coordinated recovery** in the task planner (on by default in the Studio
form). The Python equivalent is `manipulation.enable_recovery()` after enabling
adaptive grasp; API clients can set `recovery_behaviors: true` when planning.

- A free helper searches grasp sites along the object and approaches along a
  collision-checked path while the retaining arm stays fixed. Sustained contact
  and a friction/load margin are required before unloading and regrasping.
- Material warnings or damage select a registered support. If none is below the
  object, recovery searches nearby supports and elevated transport routes, then
  descends slowly. The controller unloads the jaws, verifies
  that the support carries at least 80% of the object's weight, then opens and
  checks that the object remains settled. Damage is preserved.
- A put-down ends the episode as `recovered_put_down`, not successful completion of
  the original handover. A newly recruited helper ends it as `recovered_regrasp`,
  with both grasps retained for replanning. An already-contacting helper can resume
  the saved operation. Recovery cannot bypass joint limits, collision guards,
  hard force limits or the broken-object stop.

Run the reproducible physical benchmarks:

```bash
.venv/bin/python examples/grasp_recovery.py --scenario regrasp
.venv/bin/python examples/grasp_recovery.py --scenario put_down
.venv/bin/python examples/grasp_recovery.py --scenario warning
.venv/bin/python examples/recovery_search.py --scenario helper
.venv/bin/python examples/recovery_search.py --scenario support
```

The damage benchmark seeds a known 2% bond flaw after lifting; it tests recovery
from detected pre-existing damage. The slip benchmark changes one pad's friction
and moves that tool tangentially while the helper remains steady.

Search uses the supplied collision world: pass `collision_world=world` to
`session.advance()` (Studio supplies it automatically). Studio reports checked
edges, route progress and rejected candidates. The new benchmarks use top-down
Cartesian grippers with explicit collision geometry and native structural contact.

Current limits: finite candidate search keeps measured tool orientations, samples
three bar grasp sites, and considers up to four registered supports within 0.5 m.
Elevated and lateral detours are tested; this is not a complete motion planner or
vision-based support discovery. It requires an object still retained by a gripper
and does not catch falling/fractured objects. If no checked route is available,
recovery reports the reason and does not authorize release.

### Camera-derived recovery (experimental)

Enable **Camera-derived recovery** under Coordinated recovery in Studio, or set
`camera_recovery: true` in a manipulation task request. Each captured head stereo
RGB pair is rectified and reconstructed in world coordinates. Renderer ground-truth
depth, object meshes, material colors and predefined grasp sites are not detection
inputs. The holding tool supplies only a proprioceptive seed for classical RGB/3D
region growing. A segmentation model can supply a visible-point mask through the
Python API instead.

Grasp proposals require visible opposing surfaces and a compatible jaw width.
Landing proposals require a contiguous observed horizontal patch, space for the
visible object footprint plus margins, and no observed obstruction above it.
Unknown depth and holes are rejected. Camera mode never falls back to the old
model-derived grasp sites or table centers. Low-texture stereo or occluded surfaces
can therefore produce **no proposal**; additional calibrated views or an RGB-D
sensor may be necessary.

For external calibrated depth cameras (optical Z in meters):

```python
from robogogogo.perception import backproject, fuse

manipulation.enable_adaptive_grasp(initial_force_n=16, damage_force_n=40)
manipulation.enable_recovery(camera_based=True)
cloud = backproject(depth, rgb, K, camera_position, camera_rotation,
                    timestamp=session.elapsed, uncertainty_m=0.005, source="rgbd")
# Fuse multiple synchronized views when the second grasp surface is occluded.
observation = manipulation.observe_recovery(fuse([cloud]), seed=tool_selection_point)
session.advance(collision_world=world)
```

Refresh observations during control; candidate search requires frames at most
0.5 simulation seconds old. Fused views must be within 30 ms. Studio masks stereo
pixels whose one-pixel disparity uncertainty exceeds 20 mm. This is an illustrative
uncertainty gate, not a calibrated error guarantee.

The existing robot/environment collision model remains the motion guard. Detected
landing patches are associated with physical surfaces only to verify support load;
their image-derived coordinates are not snapped to registered table centers.
Physics still needs environment geometry. This is not SLAM, unknown-scene collision
mapping, arbitrary-object grasp synthesis, or validated hardware autonomy.

Run the sensor and contact demos:

```bash
.venv/bin/python examples/camera_recovery.py
.venv/bin/python examples/camera_recovery.py --physical-helper
.venv/bin/python examples/camera_recovery.py --physical-support
.venv/bin/pytest tests/test_camera_recovery.py -q
```

The demos use an analytic two-camera RGB-D test renderer; the physical sensor
approximates the specimen with a box rather than rendering deformed triangles.
The placement fixture uses wide visible tables, shorter finger proxies enclosing
the contact pads, and a 5 mm collision margin. Studio uses the captured stereo RGB
pipeline instead. Recovery telemetry and episode observations retain the perception
source, uncertainty, candidate counts and the exact detected proposals.

### Active perception for recovery

Select **Dual arms + pan/tilt head · 14 joints** for a motorized head, or use the
existing rig's free wrist cameras. Enable **Camera-derived recovery**, then
**Active perception**. API tasks use `active_perception: true` together with
`camera_recovery: true` and `recovery_behaviors: true`.

When grasp/support search lacks usable evidence, recovery searches bounded camera
joint motions. It excludes the retaining arms' joints and shared ancestors, rejects
colliding paths, moves slowly, settles for 150 ms, and requests a new synchronized
observation. The task action chunk ends at the capture boundary so Studio captures
again before the retry. Old frames cannot authorize a new grasp or placement.
Up to four views are tried; missing fresh images or exhausted views block recovery
without releasing the object. Grip loss, object drift, increased damage, hard force
limits and the recovery timeout remain guards.

The next-view planner ranks single- and two-joint camera motions at two scales.
It keeps a bounded 8 mm surface sample map from calibrated observations, tests
candidate sight lines against observed occluders, and estimates measurement quality
from range, surface incidence, and depth uncertainty. Synchronized reference views
discount already-covered regions. Grasp searches prioritize segmented object
surfaces; support searches prioritize horizontal surfaces. Provisional object-envelope faces and nearby unknown probes
encourage missing-flank and boundary exploration but never become grasp or landing proposals.
Only positive predicted gains proceed to swept-path collision checks.

This is a local next-best-view approximation, not a global optimum or a calibrated
Bayesian uncertainty model. Unseen obstacles cannot be predicted from the surface
map, and sampling can miss thin geometry; the independent collision guard still
uses the robot's collision world. Missing, stale, or empty evidence blocks camera
exploration. Fixed head mounts are reported as immovable. Imported robots can use
their actual head/camera joints; retaining-arm joints remain protected. Native
pan/tilt dynamics use illustrative mass, inertia and limits.

Active Studio perception considers head stereo plus up to three simultaneous
head/wrist or wrist/wrist pairings. Pairs need overlap, valid calibration and
reliable disparity; invalid pairs contribute no geometry. It never uses renderer
depth to compensate for failed RGB stereo. External RGB-D observations can use the
same motion/capture/retry controller:

```python
manipulation.enable_recovery(camera_based=True, active_cameras=camera_configs)
# Keep feeding calibrated, timestamped clouds through observe_recovery().
# A view_capture phase requests an observation after the selected camera settles.
```

Run `.venv/bin/python examples/active_perception.py` to test a free wrist moving
from a view missing one grasp flank, capturing it, and completing native two-arm
regrasp. This uses the documented analytic RGB-D test sensor. View attempts, selected
camera, predicted gain, newly visible probe count, candidate ranking, capture time,
rejection reasons and actual joint targets are recorded. Predicted probe counts are
not measured grasp counts; fresh images must confirm useful surfaces.


### Temporal scene memory

Camera-derived recovery now combines head and wrist measurements across captures
in a bounded world-frame surface memory. `observe_recovery()` and Studio use the
same implementation. Synchronized views can still be combined with `fuse()`;
sequential views must have nondecreasing simulation timestamps and calibrated
poses in the same world frame.

Occluded and out-of-view geometry persists for up to eight seconds. Its position
uncertainty grows with time; reobservation reduces uncertainty with a floor for
correlated measurements. A measured depth ray extending past an old surface
invalidates that surface. A nearer occluder or a missing depth pixel does not.
Each fused camera keeps its own measured rays. Evidence that the selected object
moved also raises uncertainty on its still-hidden surfaces; the system does not
invent a new pose for them. Memory holds at most 4,000 surface samples and reserves
space for hidden history when fresh observations fill the budget.

Active perception discounts stale coverage and can seek another view of remembered
surfaces. Uncertain old samples stop acting as definite visual occluders. Grasp and
support proposals always use the current capture: remembered surfaces never
independently authorize contact or placement. Studio displays remembered and
invalidated sample counts and maximum position uncertainty.

```python
from robogogogo.perception import SceneMemory

memory = SceneMemory(capacity=4000, lifetime_s=8.0)
observation = memory.observe(head_cloud, seed=payload_selection_cue)
observation = memory.observe(later_wrist_cloud, seed=payload_selection_cue)
print(observation.diagnostics["scene_memory"])
memory.clear()  # Required if reusing it after resetting the clock or world frame.
```

Recovery owns a fresh memory for each new controller/task. This is geometric scene
memory with a selected-target mask, not semantic object identity tracking, SLAM,
or a learned motion predictor. It cannot know how an entirely hidden object moved;
uncertainty growth and expiration handle that lack of evidence. The bounded ray
sampling and noise model are approximations, and robot collision checks remain
independent. Run `pytest tests/test_scene_memory.py` for temporal fusion regressions.

### Read-only real-robot telemetry

Open **Hardware** from the Studio header to monitor real joint states and up to
eight camera feeds. Hardware telemetry has its own page and API; it does not drive
the simulation, enable motor commands, feed uncalibrated images into scene memory,
or change the task controller. No hardware actuation endpoint is implemented.

For ROS 2, source your ROS environment and use a Python interpreter compatible
with its `rclpy` and `sensor_msgs` installation, with robogogogo's Studio dependencies
installed. Register the topics when launching Studio:

```bash
python -m robogogogo.server --port 8765 \
  --ros-joints /joint_states \
  --ros-camera head_left=/head/left/image/compressed \
  --ros-camera head_right=/head/right/image/compressed \
  --ros-camera wrist_left=/left_wrist/image/compressed
```

Then open `http://127.0.0.1:8765/hardware/`, select `ros2`, and connect. An optional
`--ros-domain-id` selects the DDS domain. The adapter subscribes to `JointState`
and RGB `CompressedImage` PNG/JPEG topics with sensor-data QoS. Raw `Image`, compressed depth, moving-base TF, and hardware recording are not
implemented. CameraInfo validation, calibrated stereo, and acquisition-time
synchronization are available through a measured hardware profile; see the
[hardware calibration guide](docs/hardware-perception.md). ROS 2 is optional and
imported only when connecting; missing installations produce a visible error.
See the official [rclpy subscription example](https://github.com/ros2/tutorials/blob/master/rclpy_tutorials/topics/listener_qos_py.py)
and [sensor message definitions](https://index.ros.org/p/sensor_msgs/).

Manufacturer SDKs use a small trusted Python wrapper implementing
`TelemetryReader.read(timeout_s)` and `close()`. Return a `JointSample` or `None`
and a tuple of `CameraSample` values. Register a factory so each connection gets a
fresh SDK session:

```python
import uvicorn
from robogogogo.hardware import SDKBackend
from robogogogo.studio.api import create_app

# Implement MyTelemetryReader for your manufacturer's read-only API.
# read() must honor timeout_s; close() must unblock reads and release the client.
app = create_app(hardware_backends={
    "lab_robot": lambda: SDKBackend(MyTelemetryReader())
})
uvicorn.run(app, host="127.0.0.1", port=8765)
```

The wrapper must use SI units, preserve joint names/order, and provide sensor
acquisition timestamps. Optional velocity/effort stays missing rather than becoming
zero. Only the Python server can register adapter factories; the browser cannot
load arbitrary modules or choose arbitrary hardware addresses. Vendor wrappers
are trusted code: they must not enable motors, home the robot, or send commands
while opening or closing telemetry. No manufacturer-specific SDK is bundled.

Freshness uses the server's monotonic receipt clock (two-second default). Repeated
or backward source timestamps do not refresh a stream; reconnect after resetting
the robot clock. Joint and camera streams have independent stale indicators.
Acquisition timestamps can use different clock domains and do not establish
synchronization. Camera previews and metadata update independently. Lost browser
connectivity marks retained values stale; disconnect removes cached telemetry and
rejects late callbacks from the old adapter. The connection belongs to the server,
so closing the browser page does not disconnect other observers.

To preview this interface without hardware:

```bash
.venv/bin/python examples/hardware_monitor.py --port 8773
```

Select `demo_sdk` on `/hardware/`. Its labeled synthetic joint values and two test
camera images exercise the same SDK adapter and API. They are not robot measurements.
Run `pytest tests/test_hardware.py` for contract and integration tests. Real ROS DDS
connectivity and vendor-specific behavior still require validation on the target
robot; mock tests do not establish real-hardware readiness.


### Calibrated hardware observations and scene memory

Use a measured, versioned hardware profile to fuse real camera observations with
joint history. Studio reports calibration and synchronization readiness independently
of camera preview. No simulation pose or synthetic depth is used by this path.

The [hardware calibration and synchronization guide](docs/hardware-perception.md)
covers checkerboard intrinsic calibration with held-out validation, measured mount
transforms, clock-domain requirements, ROS CameraInfo validation, profile setup,
uncertainty rejection, scene export, and audit records. Hardware remains read-only;
fixed-base kinematics is required until timestamped moving-base localization is added.
