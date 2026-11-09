"""Structural contact adapter for the articulated manipulation integrator."""

import jax.numpy as jnp

from .grasp_contact import ContactReport


def envelope(state):
    """Conservative capsule containing every structural node (including fragments)."""
    cells = state.position.reshape(-1, 8, 3)
    axis = cells[-1].mean(0) - cells[0].mean(0)
    axis = axis / jnp.maximum(jnp.linalg.norm(axis), 1e-12)
    center = state.position.mean(0)
    along = (state.position - center) @ axis
    radial = state.position - center - along[:, None] * axis
    radius = jnp.max(jnp.linalg.norm(radial, axis=1)) + 0.001
    return jnp.stack(
        (
            center + along.min() * axis,
            center + along.max() * axis,
            jnp.array([radius, 0.0, 0.0]),
            state.position.min(0),
            state.position.max(0),
        )
    )


def structural_contact_step(
    model,
    state,
    pos,
    rot,
    vel,
    omega,
    targets,
    limits,
    mus,
    mud,
    gravity,
    supports,
    ceilings,
    protection,
):
    offset = 0.2 * rot[:, :, 0]
    centers = pos + offset
    center_velocity = vel + jnp.cross(omega, offset)
    structure, report = model._step(
        state.structure,
        centers,
        rot,
        center_velocity,
        omega,
        targets,
        limits,
        gravity=gravity,
        supports=supports,
        protect=protection,
        force_ceiling=ceilings,
        mu_static=mus,
        mu_dynamic=mud,
    )
    points = structure.position.reshape(-1, 8, 3)
    # A representative frame is telemetry only: it never drives structural nodes.
    axis_y = points[-1].mean(0) - points[0].mean(0)
    axis_y /= jnp.maximum(jnp.linalg.norm(axis_y), 1e-12)
    axis_x = jnp.mean(points[:, 4:] - points[:, :4], axis=(0, 1))
    axis_x -= jnp.dot(axis_x, axis_y) * axis_y
    axis_x /= jnp.maximum(jnp.linalg.norm(axis_x), 1e-12)
    orientation = jnp.stack((axis_x, axis_y, jnp.cross(axis_x, axis_y)), axis=1)
    center = structure.position.mean(0)
    relative = structure.position - center
    inertia = jnp.sum(
        jnp.sum(relative**2, axis=1)[:, None, None] * jnp.eye(3)
        - relative[:, :, None] * relative[:, None, :],
        axis=0,
    )
    angular = jnp.linalg.solve(
        inertia + 1e-12 * jnp.eye(3),
        jnp.sum(jnp.cross(relative, structure.velocity - structure.velocity.mean(0)), axis=0),
    )
    following = state._replace(
        structure=structure,
        position=structure.position.mean(0),
        rotation=orientation,
        velocity=structure.velocity.mean(0),
        angular_velocity=angular,
        width=structure.jaw_width,
        width_velocity=structure.jaw_velocity,
        drive_width=structure.drive_width,
    )
    return following, ContactReport(
        report.normal,
        report.friction,
        report.slip_speed,
        report.sliding,
        report.wrench_force,
        report.wrench_torque + jnp.cross(offset, report.wrench_force),
        report.penetration,
        report.normal,
        report.risk,
        report.support_load,
    )
