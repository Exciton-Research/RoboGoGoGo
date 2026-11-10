"""Single-robot workflow status, independent of HTTP and browser state.

Readiness is based on implemented paths, never on a user-set 'ready' flag.
Simulation recordings and read-only telemetry do not enable physical execution.
"""

from robogogogo.data.store import EpisodeStore
from robogogogo.learning.datasets import inspect_dataset
from robogogogo.robot.profile import RobotProfile


class RobotWorkflow:
    def __init__(self, directory, *, robot=None):
        self.robot = robot or RobotProfile()
        self.episodes = EpisodeStore(directory / "episodes")

    def validate_episode(self, identifier):
        return inspect_dataset(self.episodes.archive(identifier))

    def snapshot(self, hardware):
        robot = self.robot.snapshot()
        selected = self.robot.backend is not None and hardware.get("backend") == self.robot.backend
        connected = selected and hardware.get("status") == "streaming"
        perception = hardware.get("perception", {})
        calibrated = connected and perception.get("state") == "ready"
        episodes = self.episodes.list()
        return dict(
            robot=robot,
            physical_execution_enabled=False,
            hardware=dict(
                selected=selected,
                connected=connected,
                calibrated=calibrated,
                status=hardware.get("status", "disconnected"),
            ),
            episodes=episodes,
            stages=[
                dict(
                    id="robot",
                    title="Configure the robot",
                    state="blocked" if robot["missing"] else "configured",
                    detail="Select one physical robot and map both arms, grippers and four camera streams."
                    if robot["missing"]
                    else "Deployment identity configured; hardware commissioning is still required.",
                ),
                dict(
                    id="simulation",
                    title="Rehearse in simulation",
                    state="available",
                    detail="Dual-arm reference scene with head and wrist cameras. This is not a calibrated model of your future robot.",
                ),
                dict(
                    id="data",
                    title="Collect and validate demonstrations",
                    state="partial",
                    detail=f"{len(episodes)} episode archives. Simulation capture and dataset validation are available; physical teleoperation and recording are not connected.",
                ),
                dict(
                    id="learning",
                    title="Train and evaluate a policy",
                    state="blocked",
                    detail="Dataset validation is available. A trainer, checkpoint contract and inference adapter still need integration.",
                ),
                dict(
                    id="runtime",
                    title="Deploy to the robot",
                    state="blocked",
                    detail="Hardware is read-only. Command execution, watchdogs, takeover and a policy runner are not implemented.",
                ),
                dict(
                    id="evaluation",
                    title="Review physical trials",
                    state="blocked",
                    detail="Requires physical execution and synchronized recording of actions, outcomes and interventions.",
                ),
            ],
        )
