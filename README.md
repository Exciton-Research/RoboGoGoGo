# robogogogo

A Python/JAX and Next.js development platform for **RG Mobile Duo**: one wheeled
robot with a folding torso, two seven-joint arms, head stereo and wrist cameras.
All supported simulation, recording, learning-input and runtime workflows target
this embodiment. The mechanical design is illustrative; physical hardware has
not been selected or commissioned.

## Run Studio

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[studio,dev,perception]'
npm ci --prefix studio
npm run build --prefix studio
python -m robogogogo.server
```

Open http://127.0.0.1:8765. The local Python server serves the built Next.js UI and
API. Projects and recordings live in `runs/studio/`; one server is one shared
workspace. `python -m robogogogo --headless --steps 100` runs the same robot.

## Supported workflow

- **Robot:** versioned RG Mobile Duo geometry, joint ordering, camera mounts and
  hardware mapping. The simulation has 31 coordinates, including wheel and planar
  base coordinates. Physical left/right command groups each have seven joints.
- **Simulation:** native JAX dynamics, steering and tire friction, support-margin
  monitoring, arm self-collision checks, and bounded arm/torso detour planning.
  Movement buttons plan and run joint/torso motions directly. Drive buttons also
  start immediately. Pause retains the arm queue;
  Cancel releases ownership. Individual drive and arm commands are serialized;
  **Whole-body reach** coordinates wheels, torso and both arms toward a shared
  world-frame hand-center target. It preserves initial hand spacing and checks
  self-collision/support, without environmental obstacle or payload planning.
- **Data:** capture synchronized simulated RGB and joint states, execute an
  explicit joint-delta action, capture the next observation, then save/export.
  Archives carry the robot identity and revision. Manual, rule-based and policy
  action sources remain distinct. Debug controls invalidate ongoing recordings.
- **Learning:** validate aligned transitions and the robot contract before using
  an archive. A trainer, learned checkpoint inference and real-robot evaluation
  are still missing; the reach baseline is rule-based.
- **Runtime:** coordinated seven-joint arm and gripper commands in an opt-in test
  backend, with ownership, watchdogs, stop and takeover. It cannot move hardware.
- **Studio:** workflow readiness, simulation controls, dataset access, read-only
  hardware telemetry and command testing.

Head/wrist calibration, acquisition-time joint synchronization, ROS 2 TF and scene
memory support measured hardware observations. ROS and SDK adapters remain
read-only. Real deployment still needs selected actuators, measured geometry and
limits, a commissioned command adapter, physical stop integration and validation.

## Product boundary

Alternate robot selection, arbitrary URDF import, 2D demos, standalone structural
lab and fixed-base task endpoints are removed from the product. Old numerical
fixtures live under `tests/fixtures`; they are not supported robot experiments.
Existing recordings are not deleted or relabeled. Learning rejects archives from
other embodiments or without the current robot contract; the low-level archive
reader remains available for inspecting historical data.

Grasp/contact/material algorithms retain numerical regressions, but are not yet
integrated with this mobile robot's cosmetic fingers. No handover, damage recovery,
external-obstacle avoidance or physical grasping capability is claimed for it.
Collision checks use conservative proxies and static support sampling with runtime
guards; they do not certify hardware. Tire parameters and inertias need measurement.

## Development

```sh
.venv/bin/pytest tests/test_mobile_*.py tests/test_execution*.py tests/test_workflow_layers.py -q
.venv/bin/ruff check src tests
npm run build --prefix studio
```

See [robot design and planning](docs/mobile-robot-design.md),
[command execution](docs/command-execution.md),
[hardware calibration](docs/hardware-perception.md), and
[package layers](docs/platform.md). Keep new product experiments and model inputs
on the RG Mobile Duo definition; use minimal fixtures only for mathematical tests.
