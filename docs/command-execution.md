# Dual-arm command execution

This service implements coordinated command validation, ownership, watchdogs, stop
and takeover using a deterministic **test backend**. It never connects to physical
actuators. Existing ROS 2 and SDK telemetry backends remain read-only. The execution
supervisor rejects non-test adapters until robot-specific commissioning is implemented.

## Run and exercise the test console

```bash
npm --prefix studio run build
robogogogo-studio --test-execution
```

Open `/execution/` from the workflow. Without the flag, the status endpoint reports
`disabled` and mutation endpoints return HTTP 503. Test state is separate from both
the physical telemetry session and the JAX simulation; a moving test arm does not
move either of them.

1. Arm a manual lease or a policy test lease. Both require a fresh control epoch
   and a previously acknowledged stop. The policy test lease does not run a model.
2. Send the coordinated test pose. Both six-joint arms and both grippers use the
   same two-second quintic motion clock. Watch the measured test state update.
3. Take over manually while a policy command is active. The complete group stops
   first; the old lease is revoked before a new manual lease is granted.
4. Stop all test motion. Gripper widths are held, not opened. A stop revokes the
   current lease, cancels the active command and requires fresh stationary feedback
   for both arms and both grippers before acknowledging success.
5. Turn off owner heartbeats. After one second the watchdog stops the group and
   latches a fault. Late heartbeats cannot revive it. Explicitly reset, then arm
   again. Leaving/hiding the console requests a stop; lease expiry backs up that
   best-effort browser request.

The backend stores requested force caps, but has no contact dynamics or force
sensor. Width movement is modeled; actual gripping force is not. It is suitable for
command lifecycle tests, not validating grasp strength, collision avoidance or
physical stopping performance.

## Layer responsibilities

- `robot/execution.py`: immutable group goals, feedback and a bounded adapter
  protocol. Joint units are radians, gripper width meters, and force caps newtons.
- `robot/test_execution.py`: atomic acceptance of a complete group, common
  trajectory clock, bounded event history and backend lease expiry.
- `runtime/execution.py`: a serialized owner/command state machine plus a 20 ms
  watchdog worker. It is independent of the browser and simulation clock.
- `studio/execution.py`: strict local API and explicit opt-in application lifecycle.
- `studio/app/execution`: operator console. It is not the motor controller.

A test command always includes all six joints of each arm and both gripper goals.
Missing targets, nonfinite values and bounds violations are rejected before any
backend delivery. There is no queue and no implicit command replacement: stop the
current group first. The test limits are fixture values, not future robot limits.
The backend contract requires all-or-nothing group acceptance; sequential sends to
independent arm drivers would not satisfy it without an additional coordination
and cancellation mechanism.

## Ownership and timing

One owner (`manual` or `policy`) holds a random, opaque lease token. Tokens are only
returned on successful arming/takeover and are never included in status/events.
Each lease starts command sequence numbering at 1. A command references a recently
issued feedback sequence; observations older than 250 ms are rejected. Accepted
commands receive server-generated UUIDs. Replayed or skipped sequence numbers,
old leases, and a second command during motion are rejected.

Owner heartbeats arrive every 250 ms in Studio and extend the one-second lease in
both the service and backend. Polling status does not renew a lease. Every control
operation checks expiry first. The test backend clamps trajectory progress at its
lease deadline even if the supervisor stops polling it. On real hardware this must
be implemented by an independent controller/firmware watchdog; a Python host thread
is insufficient if the process, machine or network fails.

Feedback must be finite, match the fixed arm/gripper dimensions, remain within
position/width/speed limits, advance consistently, and be no older than 250 ms.
Timestamp rollback, conflicting sequence contents and backend errors fault the
whole group. Completion must match the active command ID, reach all targets, and
report stationary arms and grippers. Motion still active 300 ms after the expected
finish triggers the completion watchdog. A fault remains latched even when the
subsequent stop succeeds; reset does not arm or replay a command.

Stop/takeover invalidate authority before calling the driver. Stop acknowledgement
requires a newer feedback sequence with stationary arms/grippers and no backend
fault. Failed or stale acknowledgement is explicitly **unconfirmed**, never reported
as a successful stop. Arming and takeover requests include the epoch observed by
the client, preventing a delayed request from overriding a newer operator stop.

## API

All commands are local Studio POST requests; the existing origin restrictions apply.
This is a localhost operator service, not a network-authenticated robot gateway.

- `GET /api/execution`: mode, owner, epoch, limits, feedback, active command, lease
  time and the most recent 128 execution events. Diagnostics are in memory only;
  they are not a durable physical action dataset.
- `POST /api/execution/arm`: `owner` and `expected_epoch`; returns lease token and state.
- `POST /api/execution/heartbeat`: `token`.
- `POST /api/execution/command`: `token`, next `sequence`, `observation_sequence`,
  `left` and `right` joint arrays, `left_gripper` and `right_gripper` objects each
  containing `width` and `force_limit`, and `duration_s` from 0.1 to 10 seconds.
- `POST /api/execution/takeover`: `expected_epoch`; stops then grants a manual token.
- `POST /api/execution/stop`: no lease required; any local operator can stop the test.
- `POST /api/execution/reset`: explicit reset after a stop/fault; no motion command.

The supervisor checks peak speed and acceleration of the quintic profile using
1.875 × distance / duration and 5.774 × distance / duration². Targets are never
silently clipped. A future controller must execute that trajectory contract or
provide its own validated trajectory constraints; matching endpoint positions alone
is not sufficient. No jerk bound or collision check is provided by this test layer.

## Before connecting a purchased robot

Integrate the actual command interfaces, gripper force semantics, measured feedback,
controller timing, limits, coordinated start/cancel behavior and stop acknowledgement.
Commission the robot's independent watchdog and hardware stop mechanisms. Define
what hold/stop means for loaded grippers and gravity-supported arms. Bound adapter
calls: a blocking SDK call can stall this host supervisor, so hardware must retain
its own stop path. The current stop request is not a hardware emergency stop.

Still needed are the robot-specific collision/workspace checks, deployment profile
binding, continuous tracking-error monitoring, real teleoperation, durable action
recording and a trained-policy runner. Test fixtures validate service behavior;
they do not demonstrate safe physical deployment.

Run `pytest tests/test_execution.py tests/test_execution_api.py` for atomic group
progress, manual takeover, replay and stale request rejection, owner/backend lease
expiry, stale feedback, unconfirmed gripper stops, stalled completion, wrong command
acknowledgement, concurrent ownership changes, API isolation and lifecycle cleanup.
