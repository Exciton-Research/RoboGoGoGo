# Articulated mobile dual-arm reference design

Robogogogo now defaults to an original mobile robot concept inspired by the
architecture described for [DYNA Taku](https://www.dyna.co/dyna-2.1): four steerable
wheels, an articulated lower body and two seven-joint arms. This is not Taku CAD
or a reconstruction of its proprietary mechanism. Our torso joint layout,
dimensions, masses, inertias and limits are illustrative engineering assumptions.

## Inspect the mechanism

Run `python -m robogogogo.server`, open `/simulation/`, and select the Robot tab.
The articulated mobile concept is the default scene. Choose Standing, Low or
Reach to plan and start posture motion. Pause before editing a new pose. Torso
joint goals allow individual torso exploration. Wheel-drive buttons command forward/reverse,
lateral motion and yaw immediately for three simulation seconds;
Brake requests zero wheel speed. Pause freezes time and clears the drive lease. Existing arm controls manipulate
only the seven joints of each arm, not the shared torso or base.

The generated [URDF](../examples/robots/mobile_dual_arm.urdf) contains the mechanism
and simple visual geometry. Studio adds cosmetic shells and gripper fingers.
The generator in `robot/mobile_design.py` is the source of truth; `mobile_home`
defines the standing posture and alternative torso configurations.

## Reference dimensions and coordinates

- Wheel radius: 0.14 m; wheelbase: 0.50 m; wheel track: 0.68 m.
- Two folding torso segments: 0.44 m each.
- Four torso joints: yaw, lower pitch, upper pitch, chest pitch.
- Two seven-joint arms with explicit shoulder attachment boundaries: three
  co-located shoulder axes, elbow flexion, and three distal rotation axes.
  The former shoulder spacer and pedestal-style housing are removed. Internal
  `waist`/`upper_roll` names are retained for checkpoint field compatibility;
  older joint poses now refer to the revised, shorter shoulder geometry.
- Pan/tilt head, head stereo pair and one camera on each wrist.
- Four wheel steering joints and four wheel rolling joints.

There are 31 simulated coordinates: 3 ideal planar base coordinates, 8 wheel
coordinates, 4 torso joints, 14 arm joints and 2 head joints. The three planar
coordinates represent a flat-ground constraint, not physical actuators. Tire forces
now drive them through coupled articulated dynamics; joint-position commands cannot
move the base or wheel joints. Their small positive
carrier inertias satisfy the native engine's articulated-body representation.

Standing, Low and Reach presets preserve the sum of torso pitch angles, keeping
the chest level. Head and wrist camera transforms follow the entire kinematic
chain, including base movement. Camera placement and calibration remain reference
values until a physical robot is selected and measured.

## What this model establishes

The native Python/JAX engine integrates the articulated mechanism and joint
controllers. Tests check topology, joint limits, torso posture geometry, camera
alignment after base/torso motion, isolated arm inverse kinematics, and project
save/load. Next.js provides inspection and posture controls.

This establishes geometry and software interfaces, not a mechanically validated
robot. In particular:

- Base motion uses a reduced-order, flat-ground swerve model. Longitudinal wheel
  slip and lateral contact velocity generate regularized tire forces, capped by
  a shared Coulomb friction circle. Wheel reaction torques act on rolling joints.
  Steering reverses rolling direction when useful and attenuates drive until aligned.
  Motor torques are limited separately from environmental contact forces.
- Normal loads use bilinear sharing across the four wheel locations. A COM-based
  zero-moment-point estimate includes translational COM acceleration but omits
  angular momentum rate. A 35 mm support reserve guards predicted substeps.
  Violations freeze the simulation before committing the control tick; this is
  a numerical guard, not a physically demonstrated recovery or emergency stop.
- No suspension, uneven terrain, wheel lift-off, body roll/pitch, impact or rollover
  dynamics are modeled. Tire stiffness (180/220 N per m/s), friction and inertias
  need experimental identification. There is no encoder odometry or navigation.
- Drive commands are bounded to 0.5 m/s translation and 0.6 rad/s yaw, with leases
  of at most five simulation seconds. Expiry requests zero wheel speed; braking
  remains friction-limited. Leases do not run on wall time and are not hardware
  watchdogs. Base arena bounds stop prediction before virtual joint stops.
- Static torso/arm goal paths are sampled for support margin, and dynamic checks
  monitor integration. Joint acceleration requests are bounded to 0.35 rad/s², with speed limited
  to 0.35 rad/s and reduced near targets using a stopping-distance estimate.
  Arm self-collision paths are checked using conservative envelopes and adaptive
  subdivision. This does not certify physical hardware or environmental clearance.
- Masses and box-approximation inertias are assumptions; there is no motor sizing,
  structural analysis, cable routing, power design or manufacturing CAD.
- Gripper fingers are cosmetic in this concept, not independently actuated contact
  geometry. Earlier contact experiments do not validate this mobile mechanism.
- Posture presets pass self-collision checks but are not hardware stability-certified. Automated task
  planning is rejected for this scene until whole-body planning is implemented.
  Legacy fixed-base handover experiments are numerical test fixtures, not product workflows.
- Real ROS/SDK telemetry remains read-only. The separate command execution test
  backend represents two seven-joint arms and two grippers. Base and torso
  hardware execution are not implemented. Its watchdogs do not constitute physical safety certification.

## Path to a physical robot

Freeze the mechanical geometry and actuator specifications first. Then replace
illustrative inertias and limits with measured/CAD values; identify and validate the real
base steering and tire model; add whole-body collision, balance and load limits;
and extend the command backend contract for base, torso, seven-joint arms and
actuated grippers. Hardware commissioning must establish measured calibration,
latencies, stop behavior and controller limits before autonomous operation.

Simulation, recording, learning and runtime can share this robot definition, but
training, deployed inference and physical actuation remain separate unfinished
integrations. The design does not imply those workflows are already complete.


## Validation and reproducibility

`tests/test_mobile_driving.py` covers force/load balance, the friction circle,
slip dissipation, reversal/turning geometry, steering alignment, driving and braking,
zero-friction wheel spin, simultaneous torso lowering and turning, command bounds,
lease expiry and atomic guard rejection. These tests validate modeled behavior,
not agreement with measured tire data.

Project checkpoints preserve friction and the COM acceleration estimate used for
load transfer. Restoring is paused with no active drive lease. The UI reports
minimum predicted support margin, estimated tire loads and longitudinal slip.
Raw URDF export only describes geometry and inertial joints: it does not encode
the tire solver, guard, drive command lease or controller, which live in the native
Python/JAX mobile session. External URDF consumers must supply their own dynamics.

## Reference-inspired arm revision

The mobile robot now uses chest-side shoulder mounts at y = ±0.35 m and
z = 0.22 m relative to the torso frame. Its home pose places elbows below the
shoulders, with the forearms reaching forward. Upper-arm and forearm lengths
remain 0.32 m and 0.28 m. The distal three-joint assembly is shortened to
0.11 m, with corresponding COM and cylindrical inertia approximations updated.
Studio renders smooth rounded sleeves, enclosed elbow caps and compact wrist
cuffs instead of exposed industrial link boxes and stacked motor housings.
These are reference-inspired proportions, not measured vendor geometry. Shells
remain presentation geometry rather than validated collision meshes. Existing
saved joint angles retain their names but produce the revised geometry.


## Mobile self-collision guard

The mobile Session checks both commanded joint paths and every predicted physics
substep before committing a control tick. This covers torso presets, arm goals,
IK solutions, recorded policy actions, restored checkpoints, and arm motion induced
by wheel driving. Unsafe goals are rejected without replacing the current goal;
unsafe predictions clear drive, pause Studio, and leave state, elapsed time and
trajectory history unchanged. Reset is required after an execution fault.

Explicit capsules cover upper arms, forearms, elbow caps, compact wrists, grippers,
chest, head, folding torso and base pods. A 12 mm separation reserve is required.
The shoulder mounts were widened to ±0.35 m because the previous pose overlapped
the chest envelope. All opposite-arm pairs and non-attached arm/body pairs remain
enabled. Named exclusions cover intentional shared volumes at shoulder sockets,
elbows and within each compact wrist. Body-to-body pairs within the fixed torso
assembly are outside this arm-focused guard; this is not complete robot collision
certification, and the approximations have not been validated against hardware CAD.

The collision oracle checks endpoints, bounds possible proxy displacement, and
adaptively subdivides each joint-space edge. An edge that cannot be certified by
the depth limit is rejected. Continuous wheel angles are supported without angle
wrapping; prismatic bounds remain finite. This checks the simulation's piecewise
joint-space motion, not unmodeled continuous-time impacts or compliant deformation.

Studio reports the closest checked capsule pair and clearance. Arm and torso goals
now use a bounded planner: try the direct path, then a collision-checked neutral
retreat, then deterministic seeded RRT-Connect. The search has a 12-second budget;
invalid endpoints or exhausted searches preserve the previous goal. Wheels, base
and head stay fixed during planning. Stop driving and let velocities settle first.

Run executes synchronized quintic references through the waypoints. Each segment
is revalidated from actual joint positions; every predicted dynamics substep still
passes the collision and dynamic support guards before it is committed. A tracking
or stability failure blocks the motion. Pause preserves the plan; Cancel discards
it and commands a hold. Driving is unavailable while a plan owns the robot.
Checkpoint saves require completion or cancellation so a queue cannot be lost.

Recorded joint-delta policy actions retain their direct-path semantics and cannot
interrupt an active plan. They are not silently replaced with detours, which would
mislabel the recorded actions. This planner covers arm/torso self-collision proxies,
not unknown environmental obstacles, carried objects, navigation, or physical safety.


### Studio control behavior

Move arm, Move torso, posture and Cartesian movement buttons plan and start execution
in one command. Wheel buttons start a three-simulation-second drive immediately.
Pause freezes simulation and clears drive; Resume continues the arm queue. Reset
clears faults and returns to home. An active arm plan disables wheel buttons until
completion or cancellation. The displayed base position and speed confirm motion.

The arm/torso controller now solves the coupled base/wheel acceleration before
computing holding torques. This compensates tire-induced inertial loads rather than
assuming a stationary base. Motor limits and independent collision/support guards
still apply; the correction does not bypass rejected states.

The three rule-based recording fixtures use checked elbow-flexion targets for this
robot. Their position-IK baseline accepts continuous wheel-coordinate bounds but
only commands arm joints. Debug movement ends the current recording; start a new
episode before collecting another rule-based run.


## Coordinated whole-body reaching

The Robot tab now offers a whole-body hand-center target. Both hand targets are
translated by the requested displacement; their initial separation is preserved
at the endpoint. Tool orientations and intermediate hand spacing are unconstrained,
so this must not be used as a rigid-object carrying controller.

Weighted position IK includes planar base coordinates, torso joints and both arms.
Head and wheel coordinates remain outside IK. Candidate paths pass adaptive
self-collision checks and sampled static support checks; bounded RRT-Connect is
available when the direct edge is blocked. Targets are limited to local changes
within one meter, with base translation/yaw bounded around the starting state.
An infeasible endpoint or exhausted search leaves the previous target intact.

Execution shares a slow quintic reference between the joint servos and a world-to-body
base tracking controller. Base movement is produced by steering and wheel torques
with tire friction, never by writing base positions. Speed is bounded to 0.18 m/s
and 0.25 rad/s. Every actual integration substep retains collision and dynamic
support validation. A segment that fails to settle within its reference duration
plus 15 simulation seconds blocks execution. Pause freezes the trajectory clock;
Cancel discards the queue and requests guarded braking. Manual wheel commands and
recorded policy deltas cannot compete with an active plan.

This is a local simulation controller, not a hardware whole-body QP/MPC controller
or navigation system. Unknown obstacles, terrain, carried loads and contacts are
not planned. Debug whole-body movement invalidates an aligned episode; NPZ physics
trajectory export remains available. Hardware command execution is still disabled.

### Hardware appearance

Camera housings and lenses now use the episode's camera mount position and optical
orientation; head stereo spacing is 64 mm in the default rig. Wrist cameras have
mounting brackets. Joint actuator endcaps, fastening rings, hub covers, tread and
service panels follow their actual link frames. These are illustrative rendering
details, not a new collision mesh, measured actuator design, or manufacturing CAD.
Camera field-of-view guidelines are visible by default and have a separate camera
toggle in the viewport; coordinate axes are controlled independently.


### Editing a gesture

Joint and torso fields edit a pending pose. A debounced read-only preview computes
kinematics and shows an outline over the current robot: green for a clear endpoint,
red for a collision/support violation. An endpoint check does not prove the motion
path is clear. Apply pose plans and executes the complete guarded path; Discard
edits restores the current goal. Pause retains draft edits. Finish or cancel any
active trajectory before applying a replacement pose, and let velocities settle.

The draft cannot move base/wheel/head coordinates or modify the live robot,
recordings, or controller state. Its outline is hidden from RGB/depth capture;
recorded observations always depict the actual simulated robot.


### Arm actuator assumptions

The mobile integration path applies `assumed-arm-drive-1` at every physics
substep. Requested motor torque is limited by the URDF effort cap and a linear
motoring torque-speed envelope. Full peak torque remains available for braking;
there is no artificial joint velocity clamp. Smooth Coulomb gearbox friction is
an external dissipative joint force, in addition to the existing viscous damping.
The nominal inverse-dynamics controller compensates estimated friction before
actuator saturation. The estimate currently equals the assumed plant value;
robustness to identification error has not been established. Recorded torque is delivered motor
torque, excluding passive friction; a control sample records the last substep,
not every intervening torque sample.

Assumed output no-load speeds are 3 rad/s for shoulder axes, 3.5 for elbows and 5
for wrists. Coulomb friction is respectively 0.12, 0.08 and 0.025 N m, smoothed over
0.02 rad/s. These are provisional output-side values, not selected motors or
measurements. Thermal limits, bus voltage, backlash, compliance and true stiction
remain unmodeled. Base, torso and head retain their previous actuator models.
Trajectory and episode NPZ exports include the revision, provenance and parameter
arrays so data collected under these assumptions can be identified.

### Shoulder and elbow assembly frames

The detailed shoulder and elbow housings are mounted to the parent link using
URDF joint origin and rest orientation. Output flanges, shaft plugs and bolt
circles belong to the child link. The paired index marks therefore show relative
joint rotation rather than rotating the entire motor housing with its output.
Meshes consume the physics link transforms; no independent animation drives them.

The stepped casings, seals and flange details represent output-side reducer and
bearing packaging. Co-located shoulder axes remain a kinematic concept: internal
motor placement, bearing loads, transmissions, cable routing and packaging
interference are not yet mechanically validated. These details do not introduce
additional mass or contacts; existing aggregate link inertias and collision
capsules remain the physical model.

Run the frame checks with
`node --experimental-strip-types --test studio/tests/armActuator.test.mjs`.
The mobile dynamics tests also verify zero-effort gravity response, inability of
position commands to move unpowered motors directly, bounded powered gravity
holding, tire traction and coordinated reaching.
