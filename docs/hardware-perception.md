# Calibrated hardware perception

See [installation](installation.md) for the local runtime and the
[illustrated hardware screen](getting-started.md#8-understand-the-hardware-screen)
for its initial disconnected state.

This path consumes real robot telemetry, camera calibration, and acquisition
clocks. It never substitutes simulation poses or generated depth. A missing or
incompatible input blocks fusion. Raw previews remain available independently.
Hardware perception owns a separate scene memory; it cannot command the robot.

## Deployment prerequisites

- A rooted URDF with named fixed, revolute, continuous, or prismatic joints.
  Mesh visuals and inertial metadata are ignored for forward kinematics. Floating
  URDF joints, mimic joints, and unknown joint positions are rejected. A moving
  humanoid supplies the root's six-degree-of-freedom world pose through the
  timestamped localization chain below; it must not declare its base fixed.
- Measured optical-camera-to-mount transforms, robot joint zero conventions, and
  either a measured fixed base-to-world transform or an external localization
  source with verified uncertainty bounds. Optical axes are +X right, +Y down, +Z forward;
  translations are meters, joint angles radians, and prismatic positions meters.
- At least one horizontal stereo pair with overlapping equal-resolution images.
  Head/wrist pairing is allowed when its actual exposure-time geometry meets these
  conditions. Configure up to four pairs and eight cameras; every configured
  camera must participate in a pair. Lack of texture or overlap yields no depth.
- Camera and encoder acquisition timestamps in one verified clock domain. Set
  `JointSample.clock_domain` and `CameraSample.clock_domain` in SDK wrappers; ROS
  subscriptions use `ros`. The domain label is a declaration, not a clock-sync
  service. Establish hardware triggering/PTP or otherwise measure clock alignment
  before setting it. Receipt time is used only for transport freshness.

## Measure lens intrinsics

Install the optional perception dependencies and capture at least 14 unique,
sharp checkerboard images at the actual deployment resolution. Vary image
position, distance, and board tilt. Specify **inner** corner counts and measured
square size (meters):

```bash
python -m robogogogo.hardware.calibrate_intrinsics \
  --columns 9 --rows 6 --square-m 0.025 \
  --output head-left-intrinsics.json captures/head-left/*.png
```

The tool fits a five-coefficient pinhole distortion model, excludes every fourth
view from intrinsic fitting, then evaluates those held-out images with fixed
intrinsics. It rejects insufficient coverage, repeated images, low pose diversity,
and excessive reprojection error. The report contains K, distortion, image hashes,
per-image RMS, held-out indices, and calibration date. This validates the fit to
those images, not absolute real-world depth accuracy. It does not estimate
hand–eye transforms or clock offsets. Measure and independently verify those.
Fisheye lenses are not supported by this pipeline.

## Versioned profile contract

Store a JSON profile alongside the deployment URDF. Unknown fields are rejected.
`python -m robogogogo.hardware.calibration /path/robot-profile.json` validates it
without connecting hardware and prints its digest.

Required top-level fields:

- `schema_version`: `1`; `revision`: your immutable calibration identifier.
- `calibrated_at`: ISO-8601 with timezone; `robot_urdf`: path relative to the profile.
- `fixed_base`: boolean; `world_frame`: the calibrated world-frame name.
- For `fixed_base: true`, `world_from_base`: rigid 4×4 transform mapping base
  coordinates into world. For `false`, omit it and supply `localization` below.
  A static fallback for a moving robot is rejected.
- `clock_domain`: shared acquisition clock, e.g. `ros` for ROS adapters.
- `joint_speed_limits`: map of every movable URDF joint name to its verified
  operating speed bound (rad/s or m/s). Joint telemetry may contain extra joints
  and arrive in a different order; required joints are matched by name.
- `max_scene_speed_m_s`: conservative bound for scene motion during stereo skew.
  This is an operating assumption, not an observed object velocity.
- `cameras`: map of stream name to camera record below.
- `stereo_pairs`: arrays of `[left_stream_name, right_stream_name]`.
- Optional `max_skew_s` (default 0.010, maximum 0.030) and `max_joint_gap_s`
  (default 0.050, maximum 0.100): acceptance limits, not achieved timing claims.

Each camera record contains:

- `width`, `height`, `intrinsics` (3×3 K), `distortion`, `distortion_model`.
  Accepted lens models: `plumb_bob` (4/5 coefficients) or
  `rational_polynomial` (8 coefficients). The calibrated resolution must match
  decoded images exactly; no silent resize or crop is performed.
- `frame_id`: exact optical frame in image headers; `mount_link`: URDF link.
- `mount_from_optical`: measured rigid 4×4 transform from optical frame into mount.
- `reprojection_error_px`: measured quality, at most 1 pixel.
- `extrinsic_uncertainty_m`: positive conservative static positional uncertainty,
  at most 0.010 m, including mount, robot calibration, and base/world uncertainty.
- `rotation_uncertainty_rad`: positive conservative static rotational uncertainty,
  at most 0.020 radians, including the full calibrated transform chain.
- `clock_domain`: same as the profile; `time_offset_s`: measured additive correction
  to acquisition time, within ±1 s. This corrects sensor latency, not unrelated
  clock epochs. `time_uncertainty_s` is the residual timing bound and must include
  clock alignment, timestamp convention, exposure duration and rolling-shutter
  readout effects. The pipeline has no per-row rolling-shutter correction; reject
  motion/exposure conditions that cannot fit the configured bounds.
- Optional `require_camera_info`: require the ROS/SDK driver's intrinsics and
  frame ID to match the saved record before fusion. Recommended for ROS cameras.

Copy K, distortion and reprojection error from the calibration report; do not copy
report-only audit fields into camera records. Import measured mount/time records
from your calibration workflow. No generic calibration file can describe an
unknown robot correctly, so this package does not activate an example profile.
Changing calibration requires restarting/reconnecting with the new profile; old
frames and old scene memory cannot be carried across revisions.

## Connect ROS 2 or an SDK

```bash
python -m robogogogo.server --port 8765 \
  --ros-joints /joint_states \
  --ros-camera head_left=/head/left/image/compressed \
  --ros-camera head_right=/head/right/image/compressed \
  --ros-camera wrist_left=/wrist/left/image/compressed \
  --ros-camera wrist_right=/wrist/right/image/compressed \
  --ros-camera-info head_left=/head/left/camera_info \
  --ros-camera-info head_right=/head/right/camera_info \
  --ros-camera-info wrist_left=/wrist/left/camera_info \
  --ros-camera-info wrist_right=/wrist/right/camera_info \
  --hardware-profile /path/robot-profile.json
```

Camera names must exactly match the profile. Binning/cropping in `CameraInfo` is
rejected until the driver supplies an adjusted full-frame calibration. Live
`CameraInfo` validates a profile; it never silently overwrites measured extrinsics.
A mismatch latches a fault and invalidates memory until reconnect.

SDK integrations pass `hardware_profiles={"lab_robot": profile}` alongside the
existing `hardware_backends` factory map to `create_app`. `CalibrationProfile.load`
loads the profile. A custom callback backend can send `CameraInfoSample` through
`sink.calibration`; an SDK reader using a deployment-locked profile can omit
`require_camera_info`. Both routes use the same synchronizer and geometry checks.

## Moving-base localization and ROS 2 TF

Set `fixed_base` to `false` and remove `world_from_base`. The `localization` object
requires all of these fields; values must come from the deployment's measured
localization performance and operating envelope:

- `tf_chain`: two to eight unique frame names, starting at `world_frame` and ending
  at the URDF root, for example `["map", "odom", "base_link"]`. Each consecutive
  parent/child edge must be published in that direction. Names have no leading `/`.
- `max_gap_s`: positive maximum dynamic interpolation bracket, at most 0.100 s.
- `max_age_s`: maximum receipt age of dynamic transforms, at most 0.500 s and no
  smaller than `max_gap_s`. Static transforms do not expire.
- `translation_uncertainty_m` and `rotation_uncertainty_rad`: positive conservative
  bounds for the composed world-to-base estimate, at most 0.050 m and 0.100 rad.
- `max_linear_speed_m_s` and `max_angular_speed_rad_s`: positive conservative bounds
  covering every dynamic edge **and** the composed base trajectory, each at most
  10 in its respective units. Include localization estimator noise in the envelope.
- `jump_translation_m` and `jump_rotation_rad`: discontinuity thresholds, positive
  and at most 0.500 m and 1.000 rad. Exceeding a speed bound also triggers a reset.

The CLI automatically subscribes to `/tf` and `/tf_static` for a moving ROS profile.
The optional `tf2_msgs` package must be available alongside `rclpy` and `sensor_msgs`.
Only the configured chain enters localization. Static subscriptions request reliable,
transient-local delivery so existing static transforms can arrive for late subscribers.
Programmatic `ROS2Backend` integrations can override `tf_topic`, `tf_static_topic`,
and must pass the profile's `localization_chain`. The CLI uses the standard topics.

This is a bounded explicit-chain TF consumer, not a general `tf2_ros.Buffer` lookup:
there is no automatic path discovery, inverse-edge lookup, latest-pose substitution,
or extrapolation. Each dynamic edge stores at most 256 samples. Linear translation
interpolation and shortest-path quaternion SLERP resolve the chain at **each camera's
corrected exposure time**. The result composes with interpolated encoders and measured
optical mounts: world-from-base × base-from-mount × mount-from-optical. Joint TF is
ignored; the URDF and encoder history are the single source for head/wrist kinematics.
At least one localization edge must be dynamic, even while the robot is stationary.

A custom SDK callback backend sends `TransformSample` to `sink.localization`, with
parent/child frames, acquisition timestamp, translation in meters, normalized
quaternion in `(x, y, z, w)` order, shared `clock_domain`, and a `static` flag. Optional
`epoch` identifies a provider's map revision; change it when relocalization changes
coordinates. The simple polling SDK reader does not produce transforms itself.

A forward pose jump, changed static transform, or changed provider epoch immediately
invalidates exported geometry, clears queued observations, and begins a new memory
epoch. Fusion resumes only with fresh, synchronized inputs in that epoch. Changed
parents, conflicting transforms at one timestamp, invalid poses, clock rollback,
and ambiguous out-of-order discontinuities latch a fault until reconnect. Missing
or stale TF temporarily blocks fusion and scene export; it never substitutes a
fixed pose. Consumers must discard cached geometry when `localization_epoch` changes.

TF supplies poses, not covariance or localization validity. Configured uncertainty
bounds are deployment assumptions, not live quality estimates. Supply poses only
while the external estimator is valid and reconnect or change the SDK epoch when a
map revision cannot be detected as a jump. Small gradual map corrections can remain
below discontinuity thresholds; this implementation does not deform or reproject old
maps. Prefer a continuous local world frame where appropriate. SLAM, visual-inertial
odometry, map optimization, and humanoid balance/walking control remain external.

## Synchronization and uncertainty behavior

Each camera gets a bounded acquisition-time queue; joint history is bounded too.
Images have a 64 MiB total queue budget. Small arrival reordering is supported;
duplicates cannot keep streams fresh. Acquisition jumps backward by more than
one second latch a clock-reset fault, even after previous frames were consumed.
Reconnect after a clock reset. Delayed packets older than this threshold are
conservatively treated as a reset, not fused.

The synchronizer selects a new multi-camera bundle within the skew bound including
timing uncertainty. Every exposure must have bracketing joint samples, with no
extrapolation and no excessive gap or speed. Continuous joints use a constrained
shortest-angle interpolation. Camera poses are computed **at each camera's own
corrected exposure time**, not at the latest robot pose. Buffers older than 0.5 s
of receipt time are discarded; camera captures more than 0.25 s behind encoder
history are rejected. These bounds do not prove absolute acquisition latency.

Images are undistorted before stereo. A conservative scalar uncertainty budget
adds stereo disparity error, static calibration error, speed-bounded pose error
from joint interpolation/timing, and bounded scene motion. Moving localization adds
composed pose uncertainty and interpolation/timing bounds, including upstream
rotation acting on downstream TF translations and camera lever arms. Translational and
rotational uncertainty also enter the triangulation error, which grows with range
and decreasing stereo baseline. Depth above 20 mm
uncertainty is discarded. This model is an engineering bound under the configured
assumptions, not a full probabilistic estimator or a guarantee of physical accuracy.
No accepted depth means no invented geometry: existing memory ages and expires.
No synchronized observations means memory is marked stale, not advanced using a
fabricated timestamp.

## Studio and downstream consumers

Hardware → **Calibration & synchronization** shows revision/digest, clock/world
frames, skew, joint/TF brackets, localization chain/health and memory epoch/reset
reason, camera mount details, rejected/dropped samples, and
memory uncertainty. Preview freshness and fusion readiness are separate states.
`GET /api/hardware/scene` exports the bounded world-frame samples, per-sample
uncertainty, confidence, last-seen times, calibration digest, and whether the current
bundle produced depth and its `localization_epoch`. It returns HTTP 409 if the
session is unconfigured, faulted, localization is unavailable/stale, or the scene
is older than 0.5 s. This is a read-only geometry API, not action authorization.

Per-connection audit directories under `runs/studio/hardware/` hold the exact
profile, URDF, digest manifest, and rotating JSONL records (10 MB × 3 files).
Records include source image hashes/timestamps, interpolated joints, optical poses,
localization edge brackets, world-from-base poses, memory epochs, rejection reasons,
and memory statistics. Images are not stored: this is provenance,
not a complete replay dataset. Archived profiles retain their original URDF path;
use the manifest's archived URDF with the `CalibrationProfile(document, robot_xml)`
constructor to reproduce its digest independently of that original path.

## Validation boundary

Run `pytest tests/test_hardware_localization.py tests/test_hardware_perception.py
tests/test_intrinsic_calibration.py tests/test_hardware.py`.
These tests include numerical camera/robot fixtures, actual OpenCV stereo and
calibration algorithms, timing rejection, uncertainty/pose-budget checks, and API
lifecycle tests. They do not replace target-hardware validation: independently
measure transforms and clock alignment, verify reconstructed stationary geometry
while moving the base, head and wrists, and test relocalization and transport loss before using the memory for any
physical control. Hardware motion remains outside this integration.

References: [ROS CameraInfo conventions](https://github.com/ros2/common_interfaces/blob/rolling/sensor_msgs/msg/CameraInfo.msg),
[ROS timestamp synchronization](https://github.com/ros2/message_filters/blob/rolling/doc/index.rst),
[OpenCV calibration](https://docs.opencv.org/4.8.0/d9/d0c/group__calib3d.html).

Static TF QoS reference: [ROS 2 static listener QoS](https://docs.ros.org/en/rolling/p/tf2_ros/generated/classtf2__ros_1_1StaticListenerQoS.html).
