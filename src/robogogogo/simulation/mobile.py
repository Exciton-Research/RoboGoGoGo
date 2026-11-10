"""Flat-ground swerve tires and conservative support-polygon monitoring.

Planar reduced-order model: no roll/pitch impacts, suspension, terrain or rollover.
Tire forces enter coupled articulated dynamics, independently of motor effort caps.
"""

import jax
import jax.numpy as jnp
import numpy as np

from .. import spatial as sp
from .dynamics import acceleration


class MobileDynamics:
    radius = 0.14
    margin = 0.035
    corners = jnp.array([[0.25, 0.34], [0.25, -0.34], [-0.25, 0.34], [-0.25, -0.34]])

    def __init__(self, model, substeps):
        self.model, self.substeps = model, substeps

        def index(name):
            return model.indices[model.names.index(name)]

        self.base = jnp.array([index(n) for n in ("base_x", "base_y", "base_link")])
        prefixes = ["front_left", "front_right", "rear_left", "rear_right"]
        self.steer = jnp.array([index(n + "_steer") for n in prefixes])
        self.wheels = jnp.array([index(n + "_wheel") for n in prefixes])
        self.driven = np.asarray(jnp.concatenate([self.base, self.steer, self.wheels]))
        self.mass = jnp.sum(model.mass)
        from ..robot.self_collision import mobile_collision_world

        self.collision_world = mobile_collision_world(model)
        self.step = jax.jit(self._step)
        self.predict = jax.jit(self._predict)
        self.center = jax.jit(self._center)
        self.static_margin = jax.jit(lambda q: self.support(q, jnp.zeros(3))[0])

    def _center(self, q):
        pose = sp.forward_kinematics(self.model, q)
        points = pose.position + jnp.einsum("nij,nj->ni", pose.rotation, self.model.com)
        return jnp.sum(points * self.model.mass[:, None], axis=0) / self.mass

    def rotation(self, q):
        yaw = q[self.base[2]]
        c, s = jnp.cos(yaw), jnp.sin(yaw)
        return jnp.array([[c, -s], [s, c]])

    def support(self, q, com_acceleration):
        """Linear inverted-pendulum ZMP estimate; ignores angular momentum rate."""
        com = self._center(q)
        gravity = -self.model.gravity[2]
        zmp_world = com[:2] - com[2] * com_acceleration[:2] / gravity
        zmp = self.rotation(q).T @ (zmp_world - q[self.base[:2]])
        margin = jnp.min(jnp.array([0.25, 0.34]) - jnp.abs(zmp))
        # Bilinear quasi-static load sharing. Valid only while inside the polygon.
        fractions = jnp.clip(1 + self.corners * zmp / jnp.array([0.25**2, 0.34**2]), 0, 2)
        loads = self.mass * gravity * fractions[:, 0] * fractions[:, 1] / 4
        return margin, loads, zmp

    def tire_forces(self, state, mu, com_acceleration):
        """Regularized longitudinal/lateral slip with a shared Coulomb friction circle."""
        rotation = self.rotation(state.q)
        velocity = rotation.T @ state.qd[self.base[:2]]
        yaw_rate = state.qd[self.base[2]]
        contact_velocity = velocity + yaw_rate * self.corners[:, ::-1] * jnp.array([-1, 1])
        angle = state.q[self.steer]
        forward = jnp.stack([jnp.cos(angle), jnp.sin(angle)], axis=1)
        lateral = jnp.stack([-jnp.sin(angle), jnp.cos(angle)], axis=1)
        slip = self.radius * state.qd[self.wheels] - jnp.sum(contact_velocity * forward, axis=1)
        side_slip = jnp.sum(contact_velocity * lateral, axis=1)
        margin, loads, zmp = self.support(state.q, com_acceleration)
        raw = jnp.stack([180 * slip, -220 * side_slip], axis=1)
        norms = jnp.linalg.norm(raw, axis=1)
        tire = raw * jnp.minimum(1, mu * loads / jnp.maximum(norms, 1e-8))[:, None]
        body_force = tire[:, :1] * forward + tire[:, 1:] * lateral
        external = jnp.zeros_like(state.q)
        external = external.at[self.base[:2]].set(rotation @ jnp.sum(body_force, axis=0))
        external = external.at[self.base[2]].set(
            jnp.sum(self.corners[:, 0] * body_force[:, 1] - self.corners[:, 1] * body_force[:, 0])
        )
        external = external.at[self.wheels].set(-self.radius * tire[:, 0])
        return external, (margin, loads, zmp, tire, slip, side_slip)

    def steering_targets(self, state, twist):
        velocity = twist[:2] + twist[2] * self.corners[:, ::-1] * jnp.array([-1, 1])
        speed = jnp.linalg.norm(velocity, axis=1)
        desired = jnp.arctan2(velocity[:, 1], velocity[:, 0])
        current = state.q[self.steer]
        # Equivalent reversed rolling direction keeps steering within +/- pi/2.
        reverse = jnp.abs(desired) > jnp.pi / 2
        desired = jnp.where(reverse, desired - jnp.sign(desired) * jnp.pi, desired)
        desired = jnp.where(speed > 1e-5, desired, current)
        speed = jnp.where(reverse, -speed, speed)
        # Never accelerate along a requested heading before steering into it.
        speed *= jnp.maximum(0, jnp.cos(desired - current)) ** 4
        return desired, speed / self.radius

    def _step(self, state, goal, twist, mu, previous_acceleration):
        return self._predict(state, goal, twist, mu, previous_acceleration)[:5]

    def _predict(self, state, goal, twist, mu, previous_acceleration):
        model = self.model

        def substep(carry, _):
            current, previous_acc = carry
            steer_goal, wheel_speed = self.steering_targets(current, twist)
            desired_acc = jnp.clip(8 * (goal - current.q) - 6 * current.qd, -0.35, 0.35)
            desired_acc = desired_acc.at[self.base].set(0)
            torque = sp.inverse_dynamics(model, current.q, current.qd, desired_acc)
            torque += model.damping * current.qd
            torque = torque.at[self.base].set(0)  # No fictitious base motors.
            torque = torque.at[self.steer].set(
                12 * (steer_goal - current.q[self.steer]) - 4 * current.qd[self.steer]
            )
            torque = torque.at[self.wheels].set(2 * (wheel_speed - current.qd[self.wheels]))
            external, diagnostics = self.tire_forces(current, mu, previous_acc)
            qdd = acceleration(
                model, current, torque, implicit_dt=model.dt, external_force=external
            )
            qd = current.qd + model.dt * qdd
            following = sp.State(current.q + model.dt * qd, qd)
            _, old_velocity = jax.jvp(self._center, (current.q,), (current.qd,))
            _, new_velocity = jax.jvp(self._center, (following.q,), (following.qd,))
            com_acc = (new_velocity - old_velocity) / model.dt
            predicted_margin = self.support(following.q, com_acc)[0]
            return (following, com_acc), (predicted_margin, torque, diagnostics, following.q)

        (following, com_acc), trace = jax.lax.scan(
            substep, (state, previous_acceleration), None, length=self.substeps
        )
        return (
            following,
            com_acc,
            jnp.min(trace[0]),
            trace[1][-1],
            jax.tree.map(lambda x: x[-1], trace[2]),
            trace[3],
        )

    def validate_goal(self, start, goal):
        if not np.allclose(
            np.asarray(start)[self.driven], np.asarray(goal)[self.driven], atol=1e-6
        ):
            raise ValueError("Use mobile drive commands for base, steering and wheel motion")
        for fraction in np.linspace(0, 1, 25):
            q = jnp.asarray(start + fraction * (goal - start))
            if float(self.static_margin(q)) < self.margin:
                raise ValueError("Torso/arm path leaves the reserved support-polygon margin")

        if not self.collision_world.edge_free(start, goal):
            clearance = self.collision_world.last_edge_clearance
            raise ValueError(
                "Self-collision guard rejected the joint path (closest checked pair: "
                f"{clearance.pair[0]} / {clearance.pair[1]}); choose a different arm or torso goal"
            )
