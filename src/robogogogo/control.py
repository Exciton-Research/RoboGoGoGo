"""Differentiable kinematic and model-based control utilities."""

import jax
import jax.numpy as jnp

from . import spatial as sp


def inverse_kinematics(
    model, target, initial_q=None, *, body=-1, local=(0.0, 0.0, 0.0), iterations=80, damping=0.03
):
    """Position-only damped least-squares IK. Returns q and final error in meters.

    Local solver; unreachable targets may retain nonzero error. No collision or
    orientation objective. iterations must be static when jitted.
    """
    if iterations < 1 or damping <= 0:
        raise ValueError("iterations and damping must be positive")
    q = model.home if initial_q is None else initial_q

    def iteration(q, _):
        error = target - sp.point_position(model, q, body, local)
        jac = sp.point_jacobian(model, q, body, local)
        delta = jac.T @ jnp.linalg.solve(jac @ jac.T + damping**2 * jnp.eye(3), error)
        q = jnp.clip(q + jnp.clip(delta, -0.15, 0.15), model.lower, model.upper)
        return q, None

    q, _ = jax.lax.scan(iteration, q, None, length=iterations)
    return q, jnp.linalg.norm(target - sp.point_position(model, q, body, local))


def computed_torque(model, state, target_q, *, kp=80.0, kd=18.0):
    """Inverse-dynamics PD control with passive damping compensation.

    Target joint angles must use the desired continuous branch; no automatic
    wrapping is applied to bounded joints. Actuator limits apply in sp.step.
    """
    desired_acceleration = kp * (target_q - state.q) - kd * state.qd
    return (
        sp.inverse_dynamics(model, state.q, state.qd, desired_acceleration)
        + model.damping * state.qd
    )


def inverse_kinematics_targets(
    model, targets, bodies, initial_q, *, iterations=80, damping=0.03, joint_mask=None
):
    """Solve simultaneous position goals, including coupled/shared ancestor joints.

    Equal-weight damped least squares; no collision or orientation objective.
    Residuals are returned per tool so callers can detect conflicting goals.
    """

    def positions(q):
        pose = sp.forward_kinematics(model, q)
        return jnp.stack([pose.position[body] for body in bodies])

    def iteration(q, _):
        error = (targets - positions(q)).reshape(-1)
        jac = jax.jacfwd(positions)(q).reshape((-1, q.size))
        if joint_mask is not None:
            jac = jac * joint_mask[None, :]
        delta = jac.T @ jnp.linalg.solve(jac @ jac.T + damping**2 * jnp.eye(error.size), error)
        return jnp.clip(q + jnp.clip(delta, -0.15, 0.15), model.lower, model.upper), None

    q, _ = jax.lax.scan(iteration, initial_q, None, length=iterations)
    return q, jnp.linalg.norm(targets - positions(q), axis=1)


def inverse_kinematics_poses(
    model, positions, rotations, bodies, initial_q, *, iterations=240, damping=0.02, joint_mask=None
):
    """Simultaneous tool position/orientation IK using a chordal rotation residual.

    Local solver; always inspect returned position/angular errors. No automatic
    collision avoidance. Rotation matrices map tool coordinates into world.
    """

    def features(q):
        pose = sp.forward_kinematics(model, q)
        return jnp.concatenate(
            (
                pose.position[jnp.array(bodies)].reshape(-1),
                0.18 * pose.rotation[jnp.array(bodies)].reshape(-1),
            )
        )

    target = jnp.concatenate((positions.reshape(-1), 0.18 * rotations.reshape(-1)))

    def iteration(q, _):
        error = target - features(q)
        jac = jax.jacfwd(features)(q)
        if joint_mask is not None:
            jac = jac * joint_mask[None, :]
        delta = jnp.linalg.solve(jac.T @ jac + damping**2 * jnp.eye(q.size), jac.T @ error)
        return jnp.clip(q + jnp.clip(delta, -0.12, 0.12), model.lower, model.upper), None

    q, _ = jax.lax.scan(iteration, initial_q, None, length=iterations)
    pose = sp.forward_kinematics(model, q)
    actual = pose.rotation[jnp.array(bodies)]
    relative = jnp.einsum("nji,njk->nik", rotations, actual)
    angles = jnp.arccos(jnp.clip((jnp.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1))
    return q, jnp.linalg.norm(pose.position[jnp.array(bodies)] - positions, axis=1), angles
