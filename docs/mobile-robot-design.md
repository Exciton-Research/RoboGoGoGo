# Articulated mobile dual-arm reference design

Robogogogo now defaults to an original mobile robot concept inspired by the
architecture described for [DYNA Taku](https://www.dyna.co/dyna-2.1): four steerable
wheels, an articulated lower body and two seven-joint arms. This is not Taku CAD
or a reconstruction of its proprietary mechanism. Our torso joint layout,
dimensions, masses, inertias and limits are illustrative engineering assumptions.

## Inspect the mechanism

Run `python -m robogogogo.server`, open `/simulation/`, and select the Robot tab.
The articulated mobile concept is the default scene. Choose Standing, Low or
Reach, then Run posture motion. Pause before editing a new pose. Torso
joint goals allow individual torso exploration. Wheel-drive buttons command forward/reverse,
lateral motion and yaw for three simulation seconds; Run starts integration and
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
  monitor integration. Joint acceleration requests are bounded to 0.35 rad/s².
  Arm self-collision paths are checked using conservative envelopes and adaptive
  subdivision. This does not certify physical hardware or environmental clearance.
- Masses and box-approximation inertias are assumptions; there is no motor sizing,
  structural analysis, cable routing, power design or manufacturing CAD.
- Gripper fingers are cosmetic in this concept, not independently actuated contact
  geometry. Earlier contact experiments do not validate this mobile mechanism.
- Posture presets pass self-collision checks but are not hardware stability-certified. Automated task
  planning is rejected for this scene until whole-body planning is implemented.
  Select the legacy dual-arm scene for the existing handover experiments.
- Real ROS/SDK telemetry remains read-only. The separate command execution test
  backend still represents two six-joint arms, and cannot drive this seven-joint
  mobile design. Its watchdogs do not constitute physical safety certification.

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

Studio reports the closest checked capsule pair and clearance. This feature avoids
unsafe motion by rejecting it; it does not yet find automatic detours, coordinate
recovery, or avoid external obstacles. The earlier coordinated mobile task planner
remains disabled until those capabilities and whole-body planning are implemented.
