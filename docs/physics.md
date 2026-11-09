# robogogogo

**A native Python/JAX physics package for 3D robotics research.**

robogogogo implements its own rigid-body dynamics, integration, compliant contacts,
robot loading, controls, and functional research environments. It does not call
MuJoCo, Bullet, PhysX, or another physics engine. Python contains all engine and
viewer logic; JAX supplies array operations, differentiation, and compilation.
JAX itself uses compiled numerical libraries—it is not an interpreter-only dependency.

Version 0.2 is a **fixed-base 3D research preview**, not yet a general-purpose
production simulator. The original planar API remains available.

## Launch the 3D workbench

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,viewer]'
python -m robogogogo
```

The Python/Matplotlib workbench provides an orbitable 3D skeleton view, six-axis
robot arm, joint-goal sliders, Cartesian position IK, run/pause/single-step/reset,
end-effector trails, simulation telemetry, and trajectory export. Drag the scene
to orbit. Goals drive a torque controller; they do not teleport the robot.

```bash
python -m robogogogo --scene drop                   # sphere/plane contact experiment
python -m robogogogo --headless --steps 300          # six-axis controller verification
python -m robogogogo --headless --export runs/arm.npz
python -m robogogogo --urdf src/robogogogo/assets/arm6.urdf
python examples/batched_3d.py                    # randomized, batched RL collection
pytest -q
```

The viewer needs a desktop Matplotlib backend. Headless simulation needs only
JAX and NumPy. `--dt` sets physics seconds per step; `--substeps` sets physics
steps per control tick. The CLI enables float64 explicitly. Library imports do
not change global JAX settings.

## Native 3D physics API

```python
import jax
import jax.numpy as jnp
from robogogogo import spatial as physics, models

model = models.arm6()
state = physics.initial_state(model)
state = physics.step(model, state, jnp.zeros(6), substeps=5)
poses = physics.forward_kinematics(model, state.q)  # link positions + 3x3 rotations
matrix = physics.mass_matrix(model, state.q)
final, trajectory = jax.jit(physics.rollout)(model, state, jnp.zeros((100, 6)))
```

- **Topology:** arbitrary fixed-base trees, including fixed descendants and branches.
- **Joints:** revolute, continuous, and prismatic; one coordinate per moving joint.
- **Dynamics:** world-frame recursive Newton–Euler inverse dynamics. The dense mass
  matrix is formed from inverse-dynamics responses, then solved for acceleration.
- **Mass properties:** full 3×3 COM inertia tensors, rotated inertial frames, COM offsets.
- **Passive forces:** viscous damping and compliant joint stops. Actuator effort is clipped.
- **Integration:** semi-implicit position update with implicit damping and a locally
  linearized active joint-stop solve. Coriolis terms and plane contacts are explicit.
- **Contacts:** optional sphere collision proxies against a static z=0 plane, with
  unilateral spring/damper normal forces and regularized Coulomb friction applied
  at sphere surface points. Sphere rotation contributes to friction torque.
- **JAX:** immutable model/state PyTrees; `jit`, `vmap`, `scan`, and autodiff.

`acceleration()` returns continuous-time forward dynamics by default. The stepper
uses its `implicit_dt` option internally for damping and stop stability. Joint
stops and contacts permit finite penetration; they are not exact constraints.
Differentiation is piecewise at activation and clipping boundaries.

Coordinates are right-handed, z-up. A joint origin and fixed RPY orientation are
relative to its parent link; the joint axis is in that joint frame. Inertia is
about the COM, expressed in the link frame. Units are m, kg, s, rad, N and N m.
Revolute efforts are torques; prismatic efforts are forces.

## Define a robot in Python

```python
import numpy as np
from robogogogo.spatial import ModelBuilder

builder = ModelBuilder(dt=0.001)
shoulder = builder.add_link(
    'shoulder', axis=(0, 1, 0), origin=(0, 0, 0.5),
    mass=1.0, com=(0.2, 0, 0),
    inertia=np.diag([0.001, 0.014, 0.014]),
    limits=(-2.5, 2.5), effort=20.0,
)
builder.add_link(
    'tool', parent=shoulder, joint='fixed', origin=(0.4, 0, 0),
    mass=0.1, inertia=np.eye(3)*0.0001,
)
builder.add_sphere(shoulder, center=(0.4, 0, 0), radius=0.03)
model = builder.build()
```

The host-side builder validates topology, finite parameters, unit axes, positive
mass, symmetric positive-definite inertia, and the inertia triangle inequality.
Moving links require positive mass/inertia. Massless fixed frames are supported.
Parents must be added before children. Build outside `jit`, then simulate inside it.
The bundled arm is a generic model, not a calibrated model of a commercial robot.

## Load URDF

```python
from robogogogo import urdf
robot = urdf.load('src/robogogogo/assets/arm6.urdf')
model = robot.model
print(robot.joint_names)
```

Supported: one rooted fixed-base tree, fixed/revolute/continuous/prismatic joints,
joint transforms and axes, inertial-frame rotations, full inertia tensors,
effort/position limits, damping, sphere collisions, and primitive visual metadata.
All descendants require explicit inertia. The fixed root may be massless.

Unsupported floating/planar/mimic joints, meshes, nonspherical collision geometry,
transmissions, Gazebo extensions, and joint dry friction raise errors. URDF joint
velocity limits are **not enforced**. Materials and visual geometry are not rendered
by the skeleton viewer. An imported model starts at zero clipped to position limits;
URDF does not encode the native arm demo's custom home configuration.

## Controls and RL

```python
from robogogogo import env3d, control

state, observation = env3d.reset(model, jax.random.key(42))
transition = jax.jit(env3d.step)(model, state, jnp.zeros(model.home.shape))
q_goal, residual_m = jax.jit(control.inverse_kinematics)(model, state.target)
torque = control.computed_torque(model, state.physics, q_goal)
```

Position IK uses damped least squares; it is local, position-only, and reports its
residual for unreachable targets. It does not plan collision-free motions.

The reaching environment uses the last link frame as its end effector:

- Action: one normalized effort per joint, clipped to `[-1, 1]`.
- Observation: `[q, qd, target_xyz, target_minus_tip_xyz]`, size `2*n + 6`.
- Reward: negative squared distance minus `0.001 * sum(action**2)`.
- Targets: forward-kinematic samples around home, reachable but not certified collision-free.
- Default timing: 5 physics substeps/control tick, 1,000 ticks/episode.
- `success`: within 2 cm; success does not terminate the episode.
- `terminated`: numerical failure; preserves the previous valid state and returns
  a finite -100 penalty. `truncated`: horizon reached. Reset after either flag.
- No hidden PRNG state or implicit auto-reset; `examples/batched_3d.py` shows
  explicit resets while retaining terminal transitions.

`randomization.randomize(model, key)` scales mass and inertia together and varies
damping/friction. Use `vmap` over keys for independent model batches. This package
provides an environment API, not a policy-training algorithm or Gymnasium adapter.

## Record and reproduce experiments

The export button or `--export runs/example.npz` saves a compressed NumPy archive:

```python
import numpy as np
recording = np.load('runs/example.npz', allow_pickle=False)
q = recording['q']             # [samples, joints]
qd = recording['qd']
torques = recording['torque']
times = recording['time']
```

It includes the initial state, every control sample, applied torques, targets,
all numeric model parameters, topology, joint types, timestep, substeps,
integrator identifier, JAX version, and dtype. Torque row `i` was applied during
sample interval `(i-1, i)`. Simulation stores every sample in memory until reset;
long-running experiments should use their own streaming recorder. Replay is
validated on the same backend/dtype; cross-hardware bitwise identity is not promised.

## Validation and practical limits

The tests compare recursive dynamics with an independently derived Cartesian
kinetic-energy/Lagrange formulation on a mixed branched tree. They also cover
analytic pendulum motion, inertia symmetry/positivity, gravity/potential
consistency, energy convergence, contact equilibrium, friction torque, contact
and rollout gradients versus finite differences, stiff stops, URDF errors,
controller tracking, batching, environment failure handling, and trajectory replay.

`examples/batched_3d.py` reports compilation separately from synchronized warm
execution, alongside backend, dtype, reward and numerical termination counts.
Results describe that workload only, not a universal engine throughput claim.

This is an independently implemented foundation, with known boundaries:

- No floating base, free rigid bodies, quaternion joints, or closed-loop constraints.
- No general convex/mesh or complementarity contact solver. Coordinated tasks
  use primitive collision guards; manipulation adds the finger/capsule contacts below.
- No cameras/depth sensors, ROS bridge, hardware calibration, or sim-to-real validation.
- Dense mass-matrix construction is intended for modest robots; a recursive
  articulated-body forward solver is the next scalability milestone.
- Explicit compliant contacts still require timestep/stiffness studies. Implicit
  joint stops improve robustness but do not guarantee stability for arbitrary models.

A broader research product needs these capabilities plus long-horizon validation,
regression benchmarks, packaging/CI across platforms, and documented model calibration.
The next engine milestones are articulated-body forward dynamics, floating-base
state, general collision geometry, and a constraint solver, followed by sensor and
robotics ecosystem integrations.

## Original 2D examples

The original API under `robogogogo.physics` and `robogogogo.env` remains available:

```bash
python examples/reach.py
python examples/reach.py --headless
python examples/batched_rl.py
```

## Finger contact and friction

`simulation/grasp_contact.py` implements the native JAX contact kernel;
`manipulation.py` couples it to articulated dynamics and supervises grasp commands.
The object state is integrated regardless of the task's owner label. No transform
constraint overwrites object position, velocity or orientation.

For pad sample penetration δ and separating normal speed vₙ, the unilateral
normal force is `max(0, kₙ δ − cₙ vₙ)` while δ > 0. Flat pads use their plane
normal. The tangential displacement history is projected into the contact plane
and integrated from relative contact velocity. Its elastic/damping trial force is
retained inside `μ_static Fₙ`; otherwise the return map projects it onto
`μ_dynamic Fₙ` and updates the elastic history. Separated samples exert no force
and retain no tangential memory. This is a finite sampled patch approximation;
it does not implement a mesh pressure field or a complementarity solver.

Object forces and torques drive Newton–Euler translation and rotation, including
the gyroscopic term. Equal and opposite tool wrenches map through body Jacobians
to generalized joint forces. At each native 2 ms step, inertia and Jacobians are
refreshed; motor, gravity, damping and contact terms then advance on the same
0.2 ms substep grid. Both motor and contact terms use the same effective inertia
for implicit joint damping. Contact-load feedforward compensates measured loads
in the joint controller; it does not eliminate the dynamic reactions or bypass
motor effort limits. The two 0.1 kg fingers have an effective width-coordinate
mass of 0.05 kg; their motor command is rate limited to 0.15 m/s and force limited
(default 40 N per jaw). Finger inertia is represented only in the opening
coordinate, not as added articulated link mass.

Each support uses top-footprint endpoints plus an exact closest segment/AABB
point, permitting contact at an edge during tipping. Ground contact samples the
capsule axis ends and midpoint. These use compliant contact and friction too;
there is no support-height snapping in the manipulation path. Support samples
are an approximation for simultaneous contacts and do not represent contact
pressure continuously. The old `free_object_step` and `payload_model` utilities
remain available for legacy experiments, but the handover runtime uses neither.

Default per-sample pad parameters: normal stiffness 4000 N/m, damping 3 N·s/m,
tangential stiffness 10000 N/m, damping 2 N·s/m, μ_static 0.8, μ_dynamic 0.6.
Support parameters: 12000 N/m, 60 N·s/m, μ_static 0.7, μ_dynamic 0.5. These are
engineering defaults, not measured material data. They are exported in episodes.
Support rolling resistance adds a moment opposite angular velocity tangent to
the surface, bounded by `0.01 × radius × normal_force` and regularized below
0.02 rad/s. It vanishes without contact. This phenomenological finite-patch loss
prevents indefinite lossless rolling; it is not a calibrated deformation model.
The supported fixture and regression tests are not validation of arbitrary
embodiments, materials, impacts or timestep choices.

Background: [Drake's compliant contact discussion](https://drake.mit.edu/doxygen_cxx/group__compliant__contact.html)
and [dry friction model](https://drake.mit.edu/doxygen_cxx/group__friction__model.html).
These are references for the modeling choices; robogogogo does not call Drake or
another physics engine.

### Adaptive force feedback

`AdaptiveGrip` is a rule-based feedback controller over the native contact solver.
At each 10 ms control tick it filters measured tangential contact speed (20 ms
filter), detects sustained slip with hysteresis, and raises the jaw force command
at up to 80 N/s. Arm motion holds at a captured joint pose while recovering;
120 ms of low slip resumes motion. The recovered force is retained until opening.
The receiver must finish recovery before the giver releases during handover.

The command ceiling is the smaller of the actuator limit and 75% of the configured
object force threshold. Each 0.2 ms contact substep also requests opening when a
jaw load exceeds 90% of that threshold. Actual contact forces are never clipped:
peak loads across all substeps are retained, and a threshold breach latches a fault.
Persistent unrecoverable slip or lost contact blocks the task. A failed receiver
acquisition does not command the donor to release.

Studio defaults to 24 N initially per jaw for the cantilevered bar handover, with
a 40 N actuator limit and a 55 N object threshold. These are fixture parameters,
not inferred material properties. The force threshold requires calibration: this
model does not simulate deformation, contact pressure damage, or fracture and
cannot guarantee that a real object remains undamaged. Sensors here are simulated
contact patch loads and relative speed, not a hardware tactile sensor model.
Observations export controller status, force command, filtered slip, peak force,
and fault state; task metadata stores the configured thresholds and feedback rate.

Run `python examples/adaptive_grasp.py` for an articulated side-grip comparison:
friction drops after 0.8 seconds. Adaptive grip recovers; fixed 8 N grip loses
contact. This is a controlled mechanics regression, not a general grasp benchmark.

### Deformable material surfaces and damage

The optional `soft_foam`, `soft_elastomer`, and `fragile_shell` presets use a native
JAX **viscoelastic surface foundation**, distinct from the rigid reference fixture.
The presets are illustrative constitutive proxies, not measured product data.
Each has a modulus (Pa), foundation depth (m), relaxation time (s), compressive
yield/failure strains, pressure strength (Pa), plastic flow rate (1/s), and mass.
Pass a validated `Material` instance to `Manipulation(..., material=...)` to use
calibrated parameters; Studio exposes the named presets and their initial force
settings. Choosing a material also changes the object's mass and reference inertia.

A 32 × 24 axial/azimuthal grid stores indentation, permanent set, scalar damage,
current/peak pressure, and peak strain in **object coordinates**. Sliding or changing
hands does not reset the object's history. Each normal contact samples this field;
finger samples represent 24 mm² and support samples 200 mm². Pressure is force
divided by the assumed sample area. These effective areas need calibration too.
Contacts with fingers, support boxes, and the floor all contribute to deformation.

The normal pad spring/damper acts in series with a Kelvin–Voigt material element.
With indentation x, permanent set p, pad overlap δ, pad coefficients kp/cp,
material stiffness k = E A / h, viscosity η = k τ, and normal relative velocity vn,
the implicit loaded update is

```
x_next = (kp δ - cp vn + k p + (η + cp) x_old / dt)
         / (kp + k + (η + cp) / dt)
Fn = max(kp (δ - x_next) - cp (vn + (x_next - x_old) / dt), 0)
```

Unloaded cells relax toward their permanent set. A compressive viscoplastic flow
accumulates permanent indentation beyond yield; this may continue briefly during
internal stress relaxation after release. Damage is irreversible: a scalar index
increases from 0 to 1 as the larger of pressure/strength and total strain/failure
strain rises from 0.7 to 1. Stiffness falls toward 10% of its original value as
damage approaches 1. An index of 1 is a **modeled local fracture**, not a generated
crack or detached fragment. No negative normal forces or adhesive contacts are used.

The controller's risk signal additionally includes 0.7 × elastic strain/yield
strain, so it can unload before permanent yielding. Risk above 0.55 overrides
slip-driven tightening; the 0.2 ms actuator response also removes the compressive drive and brakes jaw motion while
adaptive protection is enabled. The 10 ms feedback retains the maximum risk across
its contact substeps. Any modeled damage or a permanent dent above 1 µm blocks the
coordinated task, including when adaptive force control is disabled. Faults, damage,
and peak loads latch; starting a new task creates a new, undamaged object.
These are simulator state measurements; a real robot needs a tactile/vision/material
state estimator and independently validated stopping behavior.

Studio deforms the capsule mesh from the same object-local indentation field and
colors damaged regions red. Camera RGB and rendered ground-truth depth therefore
observe the changed surface. Observations export deformation and damage fields;
task metadata exports material parameters and detection thresholds.

**Validity limits:** independent local foundation cells approximate compressible
indentation. There is no volumetric FEM, lateral strain/bulging, incompressibility,
whole-object bending, tearing, or fracture fragments. Multiple contact samples use
a maximum indentation envelope per cell; this does not solve a coupled surface PDE.
Periodic bilinear sampling avoids abrupt contact-history jumps across cells; finite grid resolution still introduces spatial discretization error. The rigid core
inertia and undeformed capsule collision envelope remain conservative approximations.
No collision margin is relaxed to make soft grasps succeed. This model is suitable
for controlled local-grasp experiments, not validated prediction of a real object's
fracture load. Background on foundation-style contact and its distinction from
elasticity solvers: [Drake's compliant contact discussion](https://drake.mit.edu/doxygen_cxx/group__compliant__contact.html).
No Drake or other physics engine is called by this implementation.

Reproducible checks:

```bash
python examples/material_grasp.py --material soft_foam
python examples/material_grasp.py --material fragile_shell
python examples/material_grasp.py --material fragile_shell --force 40 --no-adaptive
pytest tests/test_materials.py
```

The first two commands exercise compliant and fragile grasps; the third deliberately
exceeds the fragile preset's strength and should report material damage and stop.

Studio's **Material grasp → lift** recipe isolates acquisition and a level lift,
then verifies a sustained physical grasp, at least 6 cm of object elevation, low
object velocity, and no modeled damage. Selecting a deformable material chooses
this recipe by default. Delicate handling limits joint increments to 0.035 rad
per 200 ms action; the rigid reference keeps its existing motion settings.
The full cantilevered handover is a separate task: the soft-foam preset currently
can slip during transfer and is stopped by the unchanged collision guard. A
successful material grasp/lift does not certify a successful handover. Changing
grip geometry or adding distributed support is preferable to increasing damage
limits to force that task to pass.

## Structural specimens: bending, tearing and fracture

`simulation.structure.CohesiveBar` adds a **separate experimental structural model**.
The Studio **Structural Lab** (`/structure/`) runs contact-driven cantilever lift,
two-tool tearing and destructive crushing; `examples/structural_grasp.py` runs the
same recipes headlessly. Articulated manipulation can now select this model using
`Manipulation(..., structural=True)` or Studio `structural_dynamics: true`.
The local indentation model remains available for comparison.

Each hexahedral cell contains eight massive nodes and a complete graph of elastic
struts. Strut stiffness matches small axial affine-strain energy for the specified
modulus and cell volume. Compliant constraints use accumulated multipliers within
each timestep, following [XPBD](https://matthias-research.github.io/pages/publications/XPBD.pdf).
This axial calibration does not establish a continuum Poisson ratio or validate
large-deformation bending accuracy. Resolution, timestep and solver iteration count
still require convergence studies for a research application.

Four cohesive bonds join each cell interface. Their mixed-mode separation envelope
has initiation displacement `strength * cohesive_length / modulus` and terminal
displacement `2 * fracture_energy / strength`. Maximum separation and damage never
decrease. Overpressure contact work also consumes the adjacent interfaces' fracture
energy budget. This crushing rule is phenomenological: it is not a stress-tensor
crack-initiation criterion. When all four bonds fail, the interface disconnects;
nodes retain their masses and velocities, and exposed cell faces become contactable.
There is no scripted fragment explosion or object-to-gripper attachment.

Surface quadrature supplies normal penalty forces and Coulomb-limited tangential
spring forces to nodes, with equal opposite tool reaction wrenches. Finite-mass
jaws have force-limited motors. The bench prescribes tool transforms (as a robot
kinematics caller could); it does not simulate arm inertia. Destructive recipes
intentionally run without the handover damage guard. The default 0.1 ms timestep
is much smaller than the exported 20 ms observation interval.

Presets are illustrative, unmeasured specimens, not validated properties of a
specific foam, rubber or shell. Cracks follow predefined interfaces; internal cell
rupture, shell buckling, arbitrary crack propagation and full surface self-collision
are not implemented. Nodal self-contact is approximate and may miss interpenetration.
Fragments retain elastic cells; failed interfaces never heal. Centerline offset is
only a bending indicator while the specimen is connected, not curvature of fragments.
Exported JSON contains material parameters and structural trajectories, not camera
data or a robot training episode. Use float64 for numerical experiments.


### Structural feedback in articulated handover

Structural nodes, jaw widths and histories now live inside `ContactState` and are
subcycled with robot dynamics at 0.1 ms. Tool transforms and velocities come from
forward kinematics and joint Jacobians. Equal opposite structural reaction wrenches
are applied through those Jacobians to the robot; no welded payload is introduced.
Finite support tops and the floor contact the structural surface quadrature.
The physical finger patch uses 45 mm by 18 mm extents matching the Studio pads.

A simulation-state warning combines pressure relative to compressive strength with
cohesive separation plus a 5 ms positive-opening-speed forecast, relative to damage
initiation. A risk above 0.55 unloads the jaw drive and triggers adaptive unloading
and a shared arm-motion hold. Damage remains irreversible and is never reset or
clipped to make a protected trial pass. Protection reduces risk; it cannot guarantee
survival of an externally imposed impact, infeasible grasp or dropped object.
These are simulator stress signals, not a claim that a real tactile sensor measures
internal bond strain. A real robot needs an estimator and calibrated thresholds.

Structural handover adds a load-transfer phase: donor force ramps down over a second.
Warnings or an unstable receiver restore donor force while both arms hold. Intended
donor unloading does not trigger its slip-force escalation. The donor only opens
after 250 ms of stable receiver contact with donor load below 0.2 N. A blocked transfer
times out, and material damage prevents further ownership transitions.

Runtime collision checks use a conservative capsule enclosing all structural nodes,
with explicit minimum/maximum node bounds for intended support and floor contact.
Planning still uses the nominal payload; runtime guards may reject a path once the
object bends. This is not deformation-aware trajectory optimization. All fragments
remain in the collision envelope. The shared manipulation renderer supplies the same
cell meshes to Studio and camera rendering, and episode manipulation snapshots include
node positions, velocities, bonds, fragments, material risk and adaptive events.

`examples/structural_handover.py` exercises two articulated Cartesian arms, a 10 cm
lift and guarded elastomer transfer. This is a reproducible integration benchmark,
not validation of every humanoid posture or soft-material handover. Soft cantilever
grasps can still slip or fail and should not be treated as universally feasible.

### Physical recovery state machine

`GraspRecovery` consumes sustained tactile warnings, structural risk and persistent
damage. Its decision layer runs once per native control tick, while contact and
structural forces remain subcycled. It keeps the existing object state, damage,
friction histories and robot dynamics. No recovery phase welds or teleports the
object. Command reversals reset only the interpolated jaw motor target to measured
opening, avoiding an old closing target causing a squeeze after an open command.

Regrasp uses a gripper in sustained bilateral contact with a configured friction/load
margin. If none is available for a slip recovery, the planner samples three grasp
sites along the bar, rejects sites near an existing grasp, and tests open-jaw
approaches at the measured tool orientation. Masked IK freezes nonparticipating
joints, including shared joints needed by retaining arms. Each interpolated IK edge
must satisfy pose residuals and collision clearance with the live payload envelope.
Only helper fingers receive intended payload contact permission on final approach;
palms, other links and obstacles remain checked. The helper must establish stable
bilateral force and a 1.5× weight friction margin before primary unloading. After a stabilization dwell, the primary's force
ramps down; the helper must retain stable contact before the primary opens.
Reclosing and a separate verification dwell precede resuming the saved operation.
The saved operation timeout excludes time spent in recovery. Material risk or a
lost helper redirects recovery to put-down; damage is never cleared to resume.

If no support lies below the object, put-down searches up to four registered
supports within 0.5 m, nearest first. Candidate paths lift, translate, and descend
to a 50 mm standoff; higher lifts and lateral doglegs offer bounded alternatives.
All holding tools translate together at fixed orientations. Helper/support searches
have 12/15 second wall-clock budgets; these are finite candidate searches, not
complete planners. Paths run at up to 40 mm/s tool translation and 0.15 joint
units/s, with a 15 mm object-to-carrier drift rejection. Every executed physics
edge is checked again against the measured deforming envelope. Material risk pauses
route progress with hysteresis (pause above 0.55, resume below 0.45); increasing
damage blocks approach/transport without opening the retaining grasp. Support
replanning is limited to two searches per recovery attempt.

Final put-down supports vertical descent at 15 mm/s, keeping initial tool
orientations and moving the holding arms together. A simultaneous pose-IK solve
checks position/angular residuals. The selected support must be registered and
under the COM with a footprint margin. Only that support receives the existing
intentional payload-contact exemption; all other collision tests remain active.
The actual robot path and structural payload envelope are checked before committing
physics. Drift off the support, unreachable IK, lost grasps, force-limit violations,
fragmentation or a timeout (18 seconds for local recovery, 45 seconds when searching)
block recovery without authorizing release. A support's top-plane contact allowance
applies only while the payload bounds overlap its XY footprint; a departed high
support must not prevent placement on a lower nearby support.

Contact reports now expose upward support reactions (ground first, then registered
supports). After initial support loading, grip force ramps to zero. Release requires
250 ms of support load above 80% of weight with low object velocity; completion
requires another 300 ms after opening, low angular/node velocity, and negligible
jaw contact loads. Thus proximity alone cannot claim placement. A successfully
placed object ends the interrupted task for inspection. A verified intact regrasp
with an already-contacting helper can resume it. A newly recruited helper changes
the contact configuration: recovery ends as `secured`, retains both grasps, and
requires task replanning rather than replaying old waypoints. Two attempts bound
repeated retries.

Snapshots include recovery phase, reason, participants, support, attempts, support
loads, checked edge count, route progress, selected grasp and candidate rejections. Recorded task actions include `recovery_targets`, the per-tick joint goals
that override the original planner during recovery, plus all recovery/grasp events.
These are simulated contact reactions and internal material estimates, not calibrated
hardware sensing or a universal guarantee against damage during recovery.

### Camera proposal boundary

`perception.recovery` accepts calibrated visible surfaces without a dynamics model.
Backprojection uses optical camera-to-world calibration; stereo exports its rectified
intrinsics and camera rotation so rectified Z is never transformed with the original
unrectified pose. Surface normals come from neighboring valid depth samples, with
large depth discontinuities rejected. Multi-view fusion requires synchronized frames.

Classical target selection grows a color-consistent connected 3D region from a
proprioceptive tool cue. PCA and adaptive longitudinal slices propose jaw centers
only where opposing flanks are visible. The current detector is intended for elongated
specimens and fixed-orientation parallel-jaw tools. It does not infer hidden thickness
or synthesize dexterous grasps. Optional segmentation masks must label camera points,
not simulator IDs. A failed or ambiguous segmentation can cause rejection.

Horizontal surface points populate a 10 mm occupancy grid. A landing footprint must
have observed cells throughout its extent; observed obstacles above it reject it.
The selected payload itself is excluded from the obstruction check. Plane heights,
footprint sizes and centers are measured from the cloud, not enumerated from the
support registry. Association with an existing physical support is required for the
native solver's measured load verification. Unknown/mismatched supports cannot
be released onto, and a completed approach retains its selected visible footprint.

Camera mode uses fresh detections exclusively. The simulation's actual payload
state remains an independent execution/collision guard; this is intentionally not
an end-to-end perception-only world model. Real hardware would require collision
mapping, calibrated uncertainty, contact estimation and broader validation.

### Active observation loop

`ActivePerception` discovers the movable ancestors of each configured camera mount.
Closed or retaining grippers protect their entire joint ancestry, including shared
torso joints. Candidate increments are bounded by joint limits and scored by cue
alignment and transverse baseline/orientation change. The selected joint-space edge
is certified against robot/payload collision proxies; the existing simulation guard
checks executed motion again. No camera pose is edited independently of robot FK.

The controller limits exploration commands to 0.15 rad/s or 0.03 m/s and requires
joint position/velocity settling for 150 ms. `view_capture` freezes the command and
terminates the current action chunk. A distinct observation timestamped at or after
settling, no older than 0.5 s, is required to retry grasp or support planning. Two
simulation seconds without a new frame, four exhausted views, retaining-grasp loss,
15 mm object drift, increasing damage, or the existing recovery deadline stop the
attempt without opening a retaining gripper. Material risk pauses camera motion.

This is bounded geometric exploration, not learned next-best-view optimization.
A grasp requires opposing signed surface normals; same-facing edge normals cannot
substitute for an occluded flank. After successful camera exploration, manipulation
is replanned from the measured robot configuration. The optional motorized-head
fixture adds two native revolute joints; the reference arm-only fixtures retain
six/twelve joints and their original fixed mast.
