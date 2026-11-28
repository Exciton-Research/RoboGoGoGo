"""A bounded camera-feedback loop around the existing robot motion controller."""

from types import SimpleNamespace

import numpy as np

from .controller import PickupFailure, ReferenceController
from .task import HOME, TABLE_Z
from .verification import VisualVerifier


class RecoveryStopped(ValueError):
    pass


def robot_sensors(env, *, center_z):
    """Only arm/finger proprioception, robot IK, and the commanded destination."""
    return SimpleNamespace(
        data=SimpleNamespace(qpos=env.data.qpos[:9], site_xpos=env.data.site_xpos),
        site=env.site,
        dt=env.dt,
        solve=env.solve,
        target=np.r_[env.target[:2], center_z],
        previous_action=env.previous_action,
    )


class Retreat:
    def __init__(self, robot, grasp):
        tool = robot.data.site_xpos[robot.site].copy()
        raised = tool.copy()
        raised[2] = max(tool[2], grasp[2] + 0.17)
        held = 0.008 < sum(robot.data.qpos[7:9]) < 0.078
        gap = float(robot.previous_action[7]) if held else 0.08
        self.phases = [(0.9, raised, gap, False)]
        if held:
            self.phases += [
                (1.2, np.r_[grasp[:2], raised[2]], gap, False),
                (1.3, grasp + [0, 0, 0.004], gap, False),
                (0.6, grasp + [0, 0, 0.004], 0.08, False),
                (0.9, grasp + [0, 0, 0.18], 0.08, False),
            ]
        self.phases += [(1.5, HOME.copy(), 0.08, True), (0.8, HOME.copy(), 0.08, True)]
        self.phase, self.elapsed, self.done = 0, 0.0, False
        self.begin(robot)

    def begin(self, robot):
        self.start = robot.data.qpos[:7].copy()
        self.start_position = robot.data.site_xpos[robot.site].copy()
        self.elapsed = 0.0

    def action(self, robot):
        duration, goal, gap, joint_space = self.phases[self.phase]
        t = min(1, (self.elapsed + robot.dt) / duration)
        blend = t * t * t * (10 + t * (-15 + 6 * t))
        joints = (
            self.start + blend * (goal - self.start)
            if joint_space
            else robot.solve(
                self.start_position + blend * (goal - self.start_position), robot.data.qpos[:7]
            )
        )
        return np.r_[joints, gap]

    def advance(self, robot):
        self.elapsed += robot.dt
        if self.elapsed + 1e-9 < self.phases[self.phase][0]:
            return
        if self.phase == len(self.phases) - 1:
            self.done = True
        else:
            self.phase += 1
            self.begin(robot)


class ClosedLoopController:
    perception = "vlm"

    def __init__(self, robot, *, query, detection, detector, capture):
        self.query, self.detector, self.capture = query, detector, capture
        self.verifier = VisualVerifier(detector, capture)
        self.attempt, self.steps, self.done = 1, 0, False
        self.stage, self.events, self.retreat = "pickup", [], None
        self.lift_verified, self.placement_verified = False, False
        self.first_placement = None
        self._start(robot, detection)

    def _start(self, robot, detection):
        self.detection = detection
        self.pick = np.asarray(detection["position"], float)
        self.grasp = np.asarray(detection.get("grasp_position", detection["position"]), float)
        self.inner = ReferenceController(robot, camera_pick=self.pick, camera_grasp=self.grasp)
        self.inner.perception = "vlm"
        self.retreat, self.stage, self.first_placement = None, "pickup", None
        self.next_placement_check = 0.4
        self.stable_placement_samples = 0
        self._event("pickup_localized", position=self.pick.tolist(), score=detection["score"])

    def _event(self, kind, **values):
        self.events.append({"event": kind, "attempt": self.attempt, "step": self.steps, **values})

    @property
    def phase(self):
        return self.inner.phase if self.retreat is None else 0

    @property
    def elapsed(self):
        return self.inner.elapsed if self.retreat is None else self.retreat.elapsed

    @property
    def phases(self):
        return self.inner.phases if self.retreat is None else self.retreat.phases

    @property
    def labels(self):
        labels = self.inner.labels.copy()
        labels[8] = "Verify placement with cameras"
        if self.retreat is not None:
            labels[0] = "Return and release, then inspect again"
        return labels

    @property
    def feedback(self):
        return {
            "query": self.query,
            "attempt": self.attempt,
            "max_attempts": 2,
            "stage": self.stage,
            "lift_verified": self.lift_verified,
            "placement_verified": self.placement_verified,
            "events": self.events,
            "image_revision": self.verifier.image_revision,
        }

    def stop(self, reason):
        self.stage = "stopped"
        self._event("stopped", reason=str(reason))
        raise RecoveryStopped(
            f"Camera collection paused: {reason}. No more automatic attempts. Use manual control or reset."
        )

    def _recover(self, robot, reason):
        self._event("attempt_failed", reason=str(reason))
        if self.attempt >= 2:
            self.stop(reason)
        self.attempt += 1
        self.lift_verified, self.placement_verified = False, False
        self.stage = "recovering"
        self._event("retry_started", reason=str(reason))
        self.retreat = Retreat(robot, self.grasp)

    def action(self, robot):
        if self.stage == "stopped":
            raise RecoveryStopped("Automatic retry budget exhausted; use manual control or reset")
        return self.retreat.action(robot) if self.retreat is not None else self.inner.action(robot)

    def advance(self, robot):
        if self.done:
            return
        self.steps += 1
        if self.retreat is not None:
            self.retreat.advance(robot)
            if self.retreat.done:
                self.stage = "relocalizing"
                try:
                    detection = self.detector.find_target(
                        self.query,
                        self.capture,
                        workspace=(0.19, 0.84, -0.39, 0.45, TABLE_Z + 0.008, TABLE_Z + 0.23),
                    )
                    if detection["position"] is None:
                        raise ValueError(detection["pickup_error"])
                    robot.target[2] = detection["position"][2]
                    self._start(robot, detection)
                except ValueError as error:
                    self.stop(f"Fresh inspection could not support a retry: {error}")
            return
        before = self.inner.phase
        if before == 8:
            self.inner.elapsed += robot.dt
            if self.inner.elapsed + 1e-9 < self.next_placement_check:
                return
            self.next_placement_check = self.inner.elapsed + 0.5
            self.stage = "verifying_placement"
            try:
                observed = self.verifier.placement(
                    self.query,
                    destination=robot.target,
                    tool=robot.data.site_xpos[robot.site],
                    opening=float(sum(robot.data.qpos[7:9])),
                )
                if self.first_placement is None:
                    self.first_placement = observed
                    self.stable_placement_samples = 1
                    self._event("placement_observed", **observed)
                else:
                    changed_views = observed["views"] != self.first_placement["views"]
                    if changed_views or (
                        np.linalg.norm(
                            np.asarray(observed["position"]) - self.first_placement["position"]
                        )
                        > 0.008
                    ):
                        self.first_placement = observed
                        self.stable_placement_samples = 1
                        raise ValueError(
                            "Need consistent camera views to confirm settling"
                            if changed_views
                            else "The object is still moving at the destination"
                        )
                    self.first_placement = observed
                    self.stable_placement_samples += 1
                    if self.stable_placement_samples < 3:
                        self._event("placement_stable_sample", **observed)
                    else:
                        self.done, self.placement_verified, self.stage = True, True, "verified"
                        self._event("placement_verified", **observed)
            except ValueError as error:
                if self.inner.elapsed < 4.0:
                    self._event("placement_settling", reason=str(error))
                else:
                    self._recover(robot, error)
            if not self.done and self.retreat is None and self.inner.elapsed + 1e-9 >= 4.4:
                self._recover(
                    robot,
                    "Could not establish three stable observations within the settling window",
                )
            return
        try:
            self.inner.advance(robot)
        except PickupFailure as error:
            self._recover(robot, error)
            return
        if before == 3 and self.inner.phase == 4:
            self.stage = "verifying_lift"
            try:
                observed = self.verifier.lift(
                    self.query,
                    expected=robot.data.site_xpos[robot.site] + self.inner.grasp_offset,
                    opening=float(sum(robot.data.qpos[7:9])),
                )
                self.lift_verified, self.stage = True, "transport"
                self._event("lift_verified", **observed)
            except ValueError as error:
                self._recover(robot, error)
