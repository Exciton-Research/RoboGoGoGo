# One dual-arm robot, end to end

robogogogo's product workflow targets one deployment: a left arm, right arm, two
grippers, head stereo cameras, and a camera on each wrist. The hardware model has
not been selected. Empty physical mappings are intentional; the reference
simulation must never stand in for measured hardware configuration.

## Package ownership

- `robogogogo.robot`: deployment identity, ROS/SDK telemetry, calibration,
  synchronization, localization and hardware perception. Its current hardware
  contract is explicitly read-only. Simulation camera mounting utilities live
  here too; they are not physical calibration.
- `robogogogo.data`: image validation, aligned episode recording, archive storage,
  and transition readers. `EpisodeStore` owns persistence independently of Studio.
- `robogogogo.learning`: training-input inspection. `inspect_dataset` reads every
  transition, checks alignment and consistent action/camera dimensions, and returns
  the archive digest and its action specification. No trainer, checkpoint loader,
  or model inference implementation is installed by this refactor.
- `robogogogo.runtime`: shared observations/actions, deterministic policies, the
  existing simulation action supervisor, and `RobotWorkflow`. The workflow reports
  implemented capabilities; a configured profile cannot enable physical execution. The command execution
  service provides coordinated goals, watchdogs, stop and takeover with an opt-in
  test backend; physical adapters remain disabled.
- `robogogogo.studio`: HTTP/WebSocket transport and interactive simulation services.
  The Next.js application presents workflow status, simulation, hardware telemetry,
  and the episode library. It does not own a hardware motor loop or a trainer.

The numerical engine stays in `simulation` and the existing physics modules.
Planning and perception remain supporting libraries. Legacy models and material fixtures live only in tests. The product starts with
RG Mobile Duo: steerable wheels, a folding torso, two seven-joint arms and four
cameras. Other robot projects and arbitrary URDF import are no longer supported.

Robot identity, dataset readers and training-input validation can be imported
without initializing JAX or importing FastAPI. Runtime contracts have no simulator
or browser dependency. Data depends on those shared contracts, learning consumes
data, and Studio calls the services. Physical telemetry does not consume simulation
state. A future physical execution service must be distinct from the existing
simulation supervisor and from the read-only telemetry backend.

## Run the workspace

Build Studio with `npm --prefix studio run build`, then start:

```bash
robogogogo-studio --robot-profile examples/deployment/robot.json
```

The supplied profile is deliberately unconfigured. Omitting the flag gives the
same pending identity. `/` is the workflow overview, `/simulation/` is the reference
robot, and `/hardware/` is the read-only connection/calibration workspace. Only RG Mobile Duo simulation projects are supported. Robot import, structural
lab and legacy fixed-base task routes are no longer product endpoints.

When hardware is selected, fill the profile with the exact model, backend name,
disjoint ordered `left_joints`/`right_joints`, distinct gripper identifiers, and four
camera stream names. The `cameras` order is head left, head right, wrist left,
wrist right. All versioned fields must be present; unknown fields are rejected.
Restart the service to load a changed identity. The SHA-256 digest identifies that
configuration; it is not a calibration digest or a certificate of commissioning.

`--robot-profile` does not register a backend or change controller settings.
Continue using `--ros-joints`, `--ros-camera`, and `--hardware-profile` for actual
read-only ROS telemetry and measured calibration as described in
[the hardware deployment guide](hardware-perception.md). A selected backend and
matching joint/camera names are necessary for workflow calibration readiness.
Gripper identifiers are reserved configuration, not implemented gripper control.
There is no GUI for hardware mapping yet and no fake connected robot is provided.

## Working capture-to-learning-input path

1. Open the dual-arm simulation. Capture its four camera observations.
2. Execute a simulation action and capture the next observation.
3. Save/export the episode. The archive and its library entry explicitly identify
   their environment as `simulation`.
4. Open the workflow episode library and choose **Validate dataset**. Validation
   reads the saved transitions and reports counts, camera streams and SHA-256.
   Missing next observations, inconsistent dimensions, corrupt archives, and an
   archive changing during validation are rejected.
5. Consume validated data in an external training integration. Validation checks
   the data contract; it does not train a model or establish deployment suitability.

Programmatic entry points:

```python
from robogogogo.data.datasets import iter_transitions
from robogogogo.data.store import EpisodeStore
from robogogogo.learning.datasets import inspect_dataset
from robogogogo.robot.profile import RobotProfile
from robogogogo.runtime.contracts import Observation, ActionChunk, Policy
from robogogogo.runtime.supervisor import ControlSystem
```

`GET /api/workflow` reports the selected identity, hardware observation status,
episode library and stage blockers. `POST /api/data/{episode_uuid}/validate` validates
an existing local archive. These APIs neither move hardware nor launch training. The separate
[command execution test console](command-execution.md) can exercise group commands
without physical actuators using `--test-execution`.
Learning validation requires the `rg_mobile_duo` identity and the current
`ROBOT_REVISION` from `robot/definition.py` (`engineering-11` at this writing)
and 31-coordinate simulation action contract, including meter/radian units,
the `joint_coordinates` frame, and no normalization. PNG frames must have valid
checksums, decodable pixel data, and dimensions matching their camera metadata;
camera stream names must be nonempty and unique. Historical archives remain readable
through the low-level reader but are rejected as current training inputs. A simulation recording never becomes a physical demonstration because
an action is labeled `policy` or the configured robot happens to have matching DOFs.

## Integration work still required

The next deployment work must implement and validate one chosen robot's command
adapter, operator takeover, stop/watchdog behavior, and coordinated arm/gripper
execution through its supported controller. Physical demonstrations require a
teleoperation input and a streaming recorder for synchronized measured states,
commands and images. Hardware perception audit logs do not replace that recorder.

A learning integration must own dataset splits, normalization, training, checkpoint
provenance and evaluation. A physical policy runner must bind the checkpoint to the
exact robot, camera preprocessing, action representation, rate, gripper semantics
and measured calibration before connecting to execution. Current joint-delta
simulation actions do not provide that conversion. Physical trials must record
outcomes and interventions. These stages remain visibly blocked in Studio.

## Compatibility and verification

`robogogogo.hardware.*`, `vla.contracts`, `vla.recording`, `vla.images`, `vla.cameras`,
`datasets`, `system`, and `policies` remain compatibility imports of the same canonical
modules/classes. Existing `python -m robogogogo.hardware.calibration` and intrinsic
calibration commands still run. New implementation belongs in the canonical layers;
do not add a second implementation to the compatibility modules.

Run `pytest tests/test_workflow_layers.py` for module identity, lightweight imports,
strict dual-arm configuration, physical readiness isolation, and the real simulation
capture/save/validate integration test. The wider hardware, VLA, server, and control
suites protect the preserved behavior. None of these tests establishes physical
robot readiness.
