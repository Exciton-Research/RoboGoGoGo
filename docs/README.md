# robogogogo documentation

robogogogo connects a local JAX robot simulation to a browser workbench. The
supported robot is RG Mobile Duo, an illustrative mobile dual-arm design. Start
with simulation; hardware integration and learning have explicit readiness
boundaries.

## Start here

1. **[Installation](installation.md):** macOS/Linux prerequisites, scripted and
   manual setup, headless operation, updates, and troubleshooting.
2. **[Studio walkthrough](getting-started.md):** navigate the UI, move the robot,
   collect and validate an episode, and understand saved files.
3. **[Development](development.md):** run tests, edit Python/TypeScript, rebuild
   packaged assets, and regenerate the screenshots.

## Go deeper

- **[Research paper](../paper/)** provides the manuscript, reading PDF,
  bibliography, figures, and build instructions.
- **[Contributing](../CONTRIBUTING.md)** explains issues, pull requests, and licensing.
- **[Security policy](../SECURITY.md)** explains reporting and the local deployment boundary.

- **[Platform layers](platform.md)** explains robot identity, data contracts,
  runtime ownership, compatibility imports, and the remaining integration work.
- **[Mobile robot design](mobile-robot-design.md)** covers geometry, coordinate
  conventions, actuator packaging, swerve dynamics, planning, and model limits.
- **[Physics](physics.md)** documents the numerical engine and mathematical
  conventions. Some numerical algorithms are exercised only by legacy fixtures;
  their presence does not imply a Studio product feature.
- **[Command execution](command-execution.md)** explains coordinated test goals,
  leases, watchdogs, stop, takeover, and the local execution API.
- **[Hardware perception](hardware-perception.md)** describes measured calibration,
  timestamp synchronization, ROS camera topics, localization, and read-only SDK
  adapters. A simulation camera preset is not a measured calibration profile.
- **[Engineering evidence](validation/)** holds design iteration reports and
  numerical evidence. Check the robot revision when comparing old reports.

## Readiness at a glance

Simulation, browser capture, dataset validation, and command lifecycle testing
are available. Training, learned inference, physical actuation, teleoperation
recording, and physical evaluation still need integration. The workflow overview
reports these boundaries from implemented paths rather than a user-set ready flag.

![Workflow readiness and episode library](images/workflow.png)

Screenshots show the real locally built Studio with a disposable simulation
workspace. The disconnected hardware screen is intentional; no physical robot
was connected or represented as calibrated.
