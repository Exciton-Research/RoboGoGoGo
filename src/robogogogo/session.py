"""UI-independent simulation session and reproducible trajectory recorder."""

import json
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import control
from . import spatial as sp
from .embodiment import resolve_arms


class Session:
    """Deterministic simulation session shared by the GUI and headless runner."""

    def __init__(self, model, *, scene="arm", substeps=5, arms=None, mobile=False):
        self.model, self.scene, self.substeps = model, scene, substeps
        self.mobile = None
        if mobile:
            from .simulation.mobile import MobileDynamics

            self.mobile = MobileDynamics(model, substeps)
        self.drive_twist = np.zeros(3)
        self.drive_until = 0.0
        self.tire_mu = 0.7
        self.com_acceleration = jnp.zeros(3)
        self.mobile_status = None
        self.motion_plan = None
        self.motion_waypoints = []
        self.payload_model = None
        self.contact_system = None
        self.target_q = model.home
        self.state = sp.initial_state(model)
        self.elapsed = 0.0
        self.history = []
        self.arms = resolve_arms(model, arms)
        self.targets = {
            arm.name: np.asarray(sp.point_position(model, model.home, arm.body))
            for arm in self.arms
        }
        self.last_torque = np.zeros_like(model.home)
        self._advance = jax.jit(lambda state, goal: self._tick(state, goal))
        self._trace = jax.jit(self._tick_trace)
        self._loaded_trace = jax.jit(
            lambda state, goal, model: self._tick_trace(state, goal, model)
        )
        self._loaded_energy = jax.jit(sp.energy)
        self._poses = jax.jit(lambda q: sp.forward_kinematics(model, q))
        self._energy = jax.jit(lambda state: sp.energy(model, state))
        arm_mask = (
            jnp.zeros_like(model.home)
            .at[jnp.array(sorted({j for a in self.arms for j in a.joint_indices}))]
            .set(1)
        )
        self._ik = jax.jit(
            lambda targets, q: control.inverse_kinematics_targets(
                model, targets, tuple(arm.body for arm in self.arms), q, joint_mask=arm_mask
            )
        )
        # Compile outside the animation callback, without advancing visible state.
        compiled, _ = self._advance(self.state, self.target_q)
        compiled.q.block_until_ready()
        self._poses(self.state.q).position.block_until_ready()
        self._energy(self.state).block_until_ready()
        self._record()

    def poses(self) -> sp.Pose:
        """World poses for the current simulation state."""
        return self._poses(self.state.q)

    def mechanical_energy(self) -> float:
        """Rigid-body/jaw kinetic plus gravity potential energy; excludes contact springs."""
        value = float(
            self._energy(self.state)
            if self.payload_model is None
            else self._loaded_energy(self.payload_model, self.state)
        )
        if self.contact_system is not None:
            contact = self.contact_system
            if contact.structural is not None:
                nodes = contact.contact.structure
                mass = contact.structural.node_mass
                return value + float(
                    0.5 * mass * np.sum(np.asarray(nodes.velocity) ** 2)
                    - mass * np.sum(np.asarray(nodes.position) @ np.asarray(self.model.gravity))
                    + contact.structural.elastic_energy(nodes)
                    + 0.5 * 0.05 * np.sum(np.asarray(nodes.jaw_velocity) ** 2)
                )
            state, obj = contact.state, contact.object
            inertia = np.asarray(state.rotation) @ obj.inertia @ np.asarray(state.rotation).T
            value += float(
                0.5 * obj.mass * np.dot(state.velocity, state.velocity)
                + 0.5
                * np.asarray(state.angular_velocity)
                @ inertia
                @ np.asarray(state.angular_velocity)
                - obj.mass * np.dot(self.model.gravity, state.position)
                + 0.5 * 0.05 * np.sum(np.asarray(contact.contact.width_velocity) ** 2)
            )
        return value

    @property
    def target(self):
        """Compatibility alias for the first arm target."""
        return self.targets[self.arms[0].name]

    @target.setter
    def target(self, value):
        self.targets[self.arms[0].name] = np.asarray(value, dtype=float)

    def solve_targets(self, targets=None):
        values = self.targets if targets is None else targets
        return self._ik(jnp.asarray([values[arm.name] for arm in self.arms]), self.state.q)

    def solve_ik(self, target):
        goals = dict(self.targets)
        goals[self.arms[0].name] = target
        q, errors = self.solve_targets(goals)
        return q, errors[0]

    def restart_recording(self) -> None:
        """Begin a trajectory at the current state and task target."""
        self.history = []
        self._record()

    def _tick(self, state, goal):
        torque = (
            control.computed_torque(self.model, state, goal)
            if self.scene == "arm"
            else jnp.zeros_like(state.q)
        )
        return sp.step(self.model, state, torque, substeps=self.substeps), jnp.clip(
            torque, -self.model.effort, self.model.effort
        )

    def _tick_trace(self, state, goal, model=None):
        model = self.model if model is None else model
        torque = control.computed_torque(model, state, goal)

        def substep(current, _):
            following = sp.step(model, current, torque, substeps=1)
            return following, following.q

        following, positions = jax.lax.scan(substep, state, None, length=self.substeps)
        return following, jnp.clip(torque, -model.effort, model.effort), positions

    def _record(self):
        self.history.append(
            (
                self.elapsed,
                np.asarray(self.state.q).copy(),
                np.asarray(self.state.qd).copy(),
                self.last_torque.copy(),
                self.target.copy(),
                np.stack([self.targets[arm.name] for arm in self.arms]),
            )
        )

    def advance(self, *, collision_world=None):
        if self.mobile is not None:
            if (
                collision_world is not None
                or self.contact_system is not None
                or self.payload_model is not None
            ):
                raise ValueError("Mobile driving does not yet support payload/contact task models")
            return self._advance_mobile()
        if self.contact_system is not None:
            return self._advance_contacts(collision_world)
        if collision_world is None:
            if self.payload_model is None:
                following, torque = self._advance(self.state, self.target_q)
            else:
                following, torque, _ = self._loaded_trace(
                    self.state, self.target_q, self.payload_model
                )
        else:
            from .planning.collision import CollisionViolation

            if self.payload_model is None:
                following, torque, positions = self._trace(self.state, self.target_q)
            else:
                following, torque, positions = self._loaded_trace(
                    self.state, self.target_q, self.payload_model
                )
            previous = np.asarray(self.state.q)
            for q in np.asarray(positions):
                if not collision_world.edge_free(previous, q):
                    clearance = collision_world.clearance(q)
                    raise CollisionViolation(
                        f"Predicted motion violates collision margin: {clearance.pair}"
                    )
                previous = q
            # Commit only after every physics substep and connecting edge is checked.
        if (
            not np.isfinite(np.asarray(following.q)).all()
            or not np.isfinite(np.asarray(following.qd)).all()
        ):
            raise FloatingPointError(
                "Nonfinite state. Reset and reduce dt or controller/contact stiffness."
            )
        self.state = following
        self.last_torque = np.asarray(torque)
        self.elapsed += float(self.model.dt) * self.substeps
        self._record()

    def set_drive(self, twist, *, duration=3.0, friction=None):
        if self.mobile is None:
            raise ValueError("Wheel driving requires the mobile robot scene")
        if self.motion_plan and self.motion_plan["status"] in ("ready", "running", "paused"):
            raise ValueError("Finish or cancel the arm/torso plan before driving")
        value = np.asarray(twist, dtype=float)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError("Drive twist requires three finite body-frame velocities")
        if np.linalg.norm(value[:2]) > 0.5 or abs(value[2]) > 0.6:
            raise ValueError("Drive limits: translation 0.5 m/s, yaw 0.6 rad/s")
        if not np.isfinite(duration) or not 0 < duration <= 5:
            raise ValueError("Drive lease must be in (0, 5] simulation seconds")
        if friction is not None:
            if not np.isfinite(friction) or not 0 <= friction <= 1.5:
                raise ValueError("Tire friction must be in [0, 1.5]")
            self.tire_mu = float(friction)
        self.drive_twist = value
        self.drive_until = self.elapsed + duration

    def stop_drive(self):
        self.drive_twist = np.zeros(3)
        self.drive_until = self.elapsed

    def _advance_mobile(self):
        self._advance_motion_goal()
        twist = self.drive_twist if self.elapsed < self.drive_until else np.zeros(3)
        reference, velocity, reference_acceleration = self._motion_reference()
        following, acc, margin, torque, diagnostics, positions = self.mobile.predict(
            self.state,
            reference,
            jnp.asarray(twist),
            self.tire_mu,
            self.com_acceleration,
            velocity,
            reference_acceleration,
        )
        if not all(
            np.isfinite(np.asarray(v)).all() for v in (following.q, following.qd, acc, margin)
        ):
            self.stop_drive()
            raise FloatingPointError("Nonfinite mobile prediction; state was not committed")
        self.mobile_status = dict(
            support_margin_m=float(margin),
            normal_loads_n=np.asarray(diagnostics[1]).tolist(),
            zmp_body_m=np.asarray(diagnostics[2]).tolist(),
            tire_forces_n=np.asarray(diagnostics[3]).tolist(),
            longitudinal_slip_m_s=np.asarray(diagnostics[4]).tolist(),
            lateral_slip_m_s=np.asarray(diagnostics[5]).tolist(),
            friction=self.tire_mu,
            guard="ok",
        )
        if float(margin) < self.mobile.margin:
            self.stop_drive()
            self.mobile_status["guard"] = "blocked"
            raise FloatingPointError(
                "Stability guard rejected predicted motion; simulation frozen before support margin violation"
            )
        base_q = np.asarray(following.q)[np.asarray(self.mobile.base)]
        lower = np.asarray(self.model.lower)[np.asarray(self.mobile.base)]
        upper = np.asarray(self.model.upper)[np.asarray(self.mobile.base)]
        if np.any(base_q < lower + 0.05) or np.any(base_q > upper - 0.05):
            self.stop_drive()
            self.mobile_status["guard"] = "blocked"
            raise FloatingPointError(
                "Mobile arena boundary reached; simulation frozen before virtual joint stops"
            )
        from .planning.collision import CollisionViolation

        world = self.mobile.collision_world
        previous = np.asarray(self.state.q)
        closest = world.clearance(previous)
        for q in np.asarray(positions):
            clearance = world.clearance(q)
            if clearance.distance < closest.distance:
                closest = clearance
            if not world.edge_free(previous, q):
                clearance = world.last_edge_clearance
                self.stop_drive()
                self.mobile_status.update(
                    guard="blocked",
                    collision_guard="blocked",
                    collision_clearance_m=clearance.distance
                    if np.isfinite(clearance.distance)
                    else None,
                    collision_pair=list(clearance.pair),
                )
                raise CollisionViolation(
                    "Self-collision guard rejected predicted motion: "
                    f"{clearance.pair[0]} / {clearance.pair[1]}. State was not advanced."
                )
            previous = q
        self.mobile_status.update(
            collision_guard="clear",
            collision_clearance_m=closest.distance,
            collision_pair=list(closest.pair),
        )
        self.state, self.com_acceleration = following, acc
        self.target_q = self.target_q.at[self.mobile.driven].set(following.q[self.mobile.driven])
        self.last_torque = np.asarray(jnp.clip(torque, -self.model.effort, self.model.effort))
        self.elapsed += float(self.model.dt) * self.substeps
        self._record()

    def _advance_contacts(self, collision_world):
        from .planning.collision import CollisionViolation

        contact = self.contact_system
        if contact.recovery:
            contact.recovery.prepare(contact, self, collision_world)
            collision_world = contact.recovery.guarded_world(collision_world, contact)
        following, traces = contact.predict_tick(self)
        robot, object_state, _ = following
        if not all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(following)):
            raise FloatingPointError("Nonfinite contact state; reduce contact timestep/stiffness")
        if collision_world is not None:
            previous = np.asarray(self.state.q)
            segment = contact.object_segment()
            for q, current_segment in zip(np.asarray(traces[0]), np.asarray(traces[1])):
                if not collision_world.edge_free(
                    previous, q, object_start=segment, object_end=current_segment
                ):
                    clearance = collision_world.clearance(q, current_segment)
                    if contact.recovery and contact.recovery.active:
                        contact.recovery.block(
                            contact, self, "Collision guard rejected recovery motion"
                        )
                    raise CollisionViolation(
                        f"Predicted contact motion violates collision margin: {clearance.pair}"
                    )
                previous, segment = q, current_segment
        self.state = robot
        self.last_torque = np.asarray(traces[3][-1])
        self.elapsed += float(self.model.dt) * self.substeps
        contact.commit_tick(following, traces, self)
        self._record()

    def reset(self):
        self.drive_twist = np.zeros(3)
        self.drive_until = 0.0
        self.com_acceleration = jnp.zeros(3)
        self.mobile_status = None
        self.motion_plan = None
        self.motion_waypoints = []
        self.payload_model = None
        self.contact_system = None
        self.state = sp.initial_state(self.model)
        self.elapsed = 0.0
        self.last_torque = np.zeros_like(self.model.home)
        self.target_q = self.model.home
        self.targets = {
            arm.name: np.asarray(sp.point_position(self.model, self.model.home, arm.body))
            for arm in self.arms
        }
        self.history = []
        self._record()

    def set_target(self, target, arm=None):
        name = arm or self.arms[0].name
        if name not in self.targets:
            raise ValueError(f"Unknown arm: {name}")
        target = np.asarray(target, dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("target must be three finite coordinates")
        targets = dict(self.targets)
        targets[name] = target
        q, errors = self.solve_targets(targets)
        if not np.isfinite(np.asarray(q)).all():
            raise ValueError("IK failed to produce a finite configuration")
        if self.mobile is not None:
            self._plan_goal(q)
        else:
            self.target_q = q
        self.targets = targets
        return float(jnp.max(errors))

    def set_joints(self, q, arm=None):
        q = np.asarray(q, dtype=float)
        if arm is not None:
            group = next((group for group in self.arms if group.name == arm), None)
            if group is None:
                raise ValueError(f"Unknown arm: {arm}")
            if q.shape != (len(group.joint_indices),) or not np.isfinite(q).all():
                raise ValueError("joint goal must match the selected arm degrees of freedom")
            full = np.asarray(self.target_q).copy()
            full[list(group.joint_indices)] = q
            q = full
        if q.shape != self.model.home.shape or not np.isfinite(q).all():
            raise ValueError("joint goal must match model degrees of freedom")
        q = np.asarray(jnp.clip(jnp.asarray(q), self.model.lower, self.model.upper))
        if self.mobile is not None:
            self._plan_goal(q)
        else:
            self.target_q = jnp.asarray(q)
        self.targets = {
            group.name: np.asarray(sp.point_position(self.model, jnp.asarray(q), group.body))
            for group in self.arms
        }

    def _plan_goal(self, goal):
        from .planning.mobile import plan_mobile

        if np.max(np.abs(np.asarray(self.state.qd))) > 0.03:
            raise ValueError("Let the robot settle before planning a new arm/torso goal")
        if self.elapsed < self.drive_until and np.any(self.drive_twist):
            raise ValueError("Brake and settle the base before arm/torso planning")
        plan = plan_mobile(self.mobile, self.state.q, goal)
        self.stop_drive()
        self.motion_waypoints = [jnp.asarray(q) for q in plan.waypoints]
        self.motion_plan = dict(
            status="ready",
            method=plan.method,
            waypoint=1,
            waypoint_count=len(plan.waypoints) - 1,
            checks=plan.checks,
            final_goal=np.asarray(goal).tolist(),
        )
        self.target_q = self.motion_waypoints[1]
        self._begin_motion_segment()

    def _advance_motion_goal(self):
        if not self.motion_plan or self.motion_plan["status"] not in ("ready", "running", "paused"):
            return
        self.motion_plan["status"] = "running"
        driven = self.mobile.driven
        indices = np.setdiff1d(np.arange(len(self.state.q)), driven)
        error = np.max(np.abs(np.asarray(self.target_q - self.state.q)[indices]))
        speed = np.max(np.abs(np.asarray(self.state.qd)[indices]))
        if error > 0.008 or speed > 0.025:
            return
        next_index = self.motion_plan["waypoint"] + 1
        if next_index >= len(self.motion_waypoints):
            self.motion_plan["status"] = "complete"
            return
        goal = self.motion_waypoints[next_index].at[driven].set(self.state.q[driven])
        try:
            self.mobile.validate_goal(np.asarray(self.state.q), np.asarray(goal))
        except ValueError as error:
            self.motion_plan["status"] = "blocked"
            self.stop_drive()
            raise FloatingPointError(f"Planned segment failed revalidation: {error}") from error
        self.motion_plan["waypoint"] = next_index
        self.target_q = goal
        self._begin_motion_segment()

    def _begin_motion_segment(self):
        self.segment_start = self.state.q
        self.segment_time = self.elapsed
        distance = float(jnp.max(jnp.abs(self.target_q - self.segment_start)))
        self.segment_duration = max(0.5, 1.875 * distance / 0.3, (5.774 * distance / 0.2) ** 0.5)

    def _motion_reference(self):
        if not self.motion_plan or self.motion_plan["status"] not in ("ready", "running", "paused"):
            return self.target_q, None, None
        duration = self.segment_duration
        t = min(1.0, max(0.0, (self.elapsed - self.segment_time) / duration))
        s = 10 * t**3 - 15 * t**4 + 6 * t**5
        velocity = (30 * t**2 - 60 * t**3 + 30 * t**4) / duration
        acceleration = (60 * t - 180 * t**2 + 120 * t**3) / duration**2
        delta = self.target_q - self.segment_start
        return self.segment_start + s * delta, velocity * delta, acceleration * delta

    def cancel_motion_plan(self):
        self.stop_drive()
        if self.motion_plan:
            self.motion_plan["status"] = "cancelled"
        self.motion_waypoints = []
        self.target_q = self.state.q

    def export(self, destination):
        """Lossless NPZ history and complete model arrays; no pickle required.

        Row 0 is the initial state, torque row i is applied during (i-1, i).
        Export contains post-step q/qd plus targets in world coordinates.
        """
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(
            format="robogogogo-trajectory-v1",
            scene=self.scene,
            arms=[arm.to_dict() for arm in self.arms],
            substeps=self.substeps,
            integrator="semi-implicit-euler-implicit-damping-stops",
            created_utc=datetime.now(timezone.utc).isoformat(),
            parents=self.model.parents,
            kinds=self.model.kinds,
            indices=self.model.indices,
            names=self.model.names,
            jax_version=jax.__version__,
            dtype=str(self.state.q.dtype),
        )
        times, q, qd, torque, target, arm_targets = zip(*self.history)
        arrays = {
            f"model_{f.name}": np.asarray(getattr(self.model, f.name))
            for f in fields(self.model)
            if f.name not in ("parents", "kinds", "indices", "names")
        }
        # File handle preserves the requested path instead of silently adding .npz.
        with path.open("wb") as output:
            np.savez_compressed(
                output,
                metadata=np.array(json.dumps(metadata)),
                time=np.array(times),
                q=np.stack(q),
                qd=np.stack(qd),
                torque=np.stack(torque),
                target=np.stack(target),
                arm_targets=np.stack(arm_targets),
                **arrays,
            )
        return path.resolve()
