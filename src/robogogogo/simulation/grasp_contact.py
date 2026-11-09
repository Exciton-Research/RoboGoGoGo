"""Finite sampled finger patches, compliant normals and elasto-plastic Coulomb friction.

All forces are SI. This is a discrete patch approximation, not a mesh contact solver.
Tangential spring history provides stiction; yielding projects onto the friction cone.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .materials import MaterialState, material_response, surface_stencil
from .math import rotation
from .structure import StructureState


class ContactState(NamedTuple):
    position: jax.Array
    rotation: jax.Array
    velocity: jax.Array
    angular_velocity: jax.Array
    width: jax.Array
    width_velocity: jax.Array
    tangent: jax.Array
    support_tangent: jax.Array
    drive_width: jax.Array
    material_state: MaterialState | None = None
    structure: StructureState | None = None


class ContactReport(NamedTuple):
    normal: jax.Array
    friction: jax.Array
    slip_speed: jax.Array
    sliding: jax.Array
    force: jax.Array
    moment: jax.Array
    penetration: jax.Array
    peak_normal: jax.Array
    material_risk: jax.Array = 0.0
    support_load: jax.Array | None = None


def friction_force(
    normal,
    penetration,
    relative_velocity,
    memory,
    dt,
    mu_static,
    mu_dynamic,
    stiffness=4000.0,
    damping=3.0,
    tangent_stiffness=10000.0,
    tangent_damping=2.0,
    normal_force=None,
):
    vn = jnp.sum(relative_velocity * normal, axis=-1)
    active = penetration > 0
    fn = jnp.where(active, jnp.maximum(stiffness * penetration - damping * vn, 0.0), 0.0)
    if normal_force is not None:
        fn = normal_force
    vt = relative_velocity - vn[..., None] * normal
    displacement = memory + dt * vt
    displacement -= jnp.sum(displacement * normal, axis=-1)[..., None] * normal
    trial = -tangent_stiffness * displacement - tangent_damping * vt
    magnitude = jnp.linalg.norm(trial, axis=-1)
    sliding = magnitude > mu_static * fn
    # Return mapping retains elastic pre-slip displacement without unbounded windup.
    ft = trial * jnp.where(sliding, mu_dynamic * fn / jnp.maximum(magnitude, 1e-12), 1.0)[..., None]
    ft = jnp.where((fn > 0)[..., None], ft, 0.0)
    memory = jnp.where(
        sliding[..., None], -(ft + tangent_damping * vt) / tangent_stiffness, displacement
    )
    memory = jnp.where((fn > 0)[..., None], memory, 0.0)
    return (
        fn[..., None] * normal + ft,
        memory,
        fn,
        jnp.linalg.norm(ft, axis=-1),
        jnp.linalg.norm(vt, axis=-1),
        sliding & (fn > 0),
    )


def closest_segment_box_parameter(a, b, lower, upper):
    """Exact piecewise-quadratic segment/AABB minimizer (batched boxes)."""
    direction = b - a
    denom = jnp.where(jnp.abs(direction) > 1e-12, direction, 1.0)
    crossings = jnp.concatenate(((lower - a) / denom, (upper - a) / denom), axis=1)
    crossings = jnp.where(jnp.tile(jnp.abs(direction) > 1e-12, 2), crossings, 0.0)
    cuts = jnp.sort(
        jnp.concatenate(
            (
                jnp.zeros((lower.shape[0], 1)),
                jnp.ones((lower.shape[0], 1)),
                jnp.clip(crossings, 0.0, 1.0),
            ),
            axis=1,
        ),
        axis=1,
    )
    mid = (cuts[:, :-1] + cuts[:, 1:]) / 2
    points = a + mid[:, :, None] * direction
    active = (points < lower[:, None]) | (points > upper[:, None])
    boundary = jnp.where(points < lower[:, None], lower[:, None], upper[:, None])
    numerator = jnp.sum(direction * (a - boundary) * active, axis=2)
    denominator = jnp.sum(direction**2 * active, axis=2)
    roots = jnp.clip(
        jnp.where(denominator > 1e-14, -numerator / jnp.maximum(denominator, 1e-14), mid),
        cuts[:, :-1],
        cuts[:, 1:],
    )
    candidates = jnp.concatenate((cuts, roots, mid), axis=1)
    points = a + candidates[:, :, None] * direction
    outside = jnp.maximum(jnp.maximum(lower[:, None] - points, points - upper[:, None]), 0.0)
    distances = jnp.sum(outside**2, axis=2)
    choice = jnp.argmin(distances + 1e-12 * (candidates - 0.5) ** 2, axis=1)
    return jnp.take_along_axis(candidates, choice[:, None], axis=1)[:, 0]


def contact_step(
    state,
    tool_position,
    tool_rotation,
    tool_velocity,
    tool_omega,
    target_width,
    force_limit,
    mu_static,
    mu_dynamic,
    mass,
    inertia,
    radius,
    half_length,
    gravity,
    supports,
    dt,
    rolling_resistance=0.01,
    force_ceiling=None,
    material=None,
    protect_material=False,
):
    """Advance object and finite-mass jaw actuators; return equal/opposite tool wrenches.

    Each jaw has six surface samples over a 12 x 12 mm contact patch. The jaw pair
    has effective opening mass 0.05 kg. Its position servo is force limited.
    Tool motion is supplied by the articulated integrator, never an object weld.
    """
    sides = jnp.array([-1.0, 1.0])
    xs = jnp.tile(jnp.array([0.194, 0.200, 0.206]), 2)
    zs = jnp.repeat(jnp.array([-0.006, 0.006]), 3)
    shape = (state.width.size, 2, 6)
    local = jnp.stack(
        (
            jnp.broadcast_to(xs, shape),
            jnp.broadcast_to(state.width[:, None, None] * sides[None, :, None] / 2, shape),
            jnp.broadcast_to(zs, shape),
        ),
        axis=-1,
    )
    offset = jnp.einsum("aij,apkj->apki", tool_rotation, local)
    points = tool_position[:, None, None, :] + offset
    jaw_motion = tool_rotation[:, None, None, :, 1] * (
        state.width_velocity[:, None, None, None] * sides[None, :, None, None] / 2
    )
    pad_velocity = (
        tool_velocity[:, None, None, :]
        + jnp.cross(tool_omega[:, None, None, :], offset)
        + jaw_motion
    )
    axis = state.rotation[:, 1]
    along = jnp.clip(jnp.sum((points - state.position) * axis, axis=-1), -half_length, half_length)
    centers = state.position + along[..., None] * axis
    # Flat inner pad faces exert their plane normal, not a radial sphere normal.
    # Using radial normals at sample nodes would invent geometric friction.
    normal = jnp.broadcast_to(
        -sides[None, :, None, None] * tool_rotation[:, None, None, :, 1], points.shape
    )
    difference = centers - points
    signed = jnp.sum(difference * normal, axis=-1)
    tangent_distance2 = jnp.sum((difference - signed[..., None] * normal) ** 2, axis=-1)
    extent = jnp.sqrt(jnp.maximum(radius**2 - tangent_distance2, 0.0))
    penetration = jnp.where((tangent_distance2 < radius**2) & (signed >= 0), extent - signed, 0.0)
    contact = points - penetration[..., None] * normal
    lever = contact - state.position
    relative = state.velocity + jnp.cross(state.angular_velocity, lever) - pad_velocity
    # Sample the actual axis/support overlap, including both footprint edges.
    # Uniform samples on the whole bar can miss a support edge while it tips.
    xy_axis = axis[:2]
    lower = supports[:, :2] - supports[:, 2:4]
    upper = supports[:, :2] + supports[:, 2:4]
    denom = jnp.where(jnp.abs(xy_axis) > 1e-10, xy_axis, 1.0)
    ta, tb = (lower - state.position[:2]) / denom, (upper - state.position[:2]) / denom
    parallel_inside = (state.position[:2] >= lower) & (state.position[:2] <= upper)
    near = jnp.where(
        jnp.abs(xy_axis) > 1e-10, jnp.minimum(ta, tb), jnp.where(parallel_inside, -jnp.inf, jnp.inf)
    )
    far = jnp.where(
        jnp.abs(xy_axis) > 1e-10, jnp.maximum(ta, tb), jnp.where(parallel_inside, jnp.inf, -jnp.inf)
    )
    lo = jnp.maximum(-half_length, jnp.max(near, axis=1))
    hi = jnp.minimum(half_length, jnp.min(far, axis=1))
    valid = lo <= hi
    lo, hi = jnp.where(valid, lo, 0.0), jnp.where(valid, hi, 0.0)
    box_lower = jnp.concatenate((lower, jnp.zeros((supports.shape[0], 1))), axis=1)
    box_upper = jnp.concatenate((upper, supports[:, 4, None]), axis=1)
    optimum = closest_segment_box_parameter(
        state.position - half_length * axis,
        state.position + half_length * axis,
        box_lower,
        box_upper,
    )
    optimum = -half_length + 2 * half_length * optimum
    samples = jnp.stack((lo, optimum, hi), axis=1)
    centers_box = state.position + samples[:, :, None] * axis
    nearest = jnp.clip(centers_box, box_lower[:, None], box_upper[:, None])
    difference = centers_box - nearest
    distance = jnp.linalg.norm(difference, axis=2)
    normals_box = jnp.where(
        (distance > 1e-12)[:, :, None],
        difference / jnp.maximum(distance[:, :, None], 1e-12),
        jnp.array([0.0, 0.0, 1.0]),
    )
    penetration_box = radius - distance
    # The exact closest point always participates, even outside the top footprint.
    valid_box = jnp.stack((valid, jnp.ones_like(valid), valid), axis=1)
    penetration_box = jnp.where(valid_box, penetration_box, 0.0)
    centers_floor = state.position + jnp.array([-half_length, 0.0, half_length])[:, None] * axis
    support_normal = jnp.concatenate(
        (jnp.tile(jnp.array([0.0, 0.0, 1.0]), (3, 1)), normals_box.reshape(-1, 3))
    )
    centers_all = jnp.concatenate((centers_floor, centers_box.reshape(-1, 3)))
    support_points = centers_all - radius * support_normal
    support_penetration = jnp.concatenate(
        (radius - centers_floor[:, 2], penetration_box.reshape(-1))
    )
    support_lever = support_points - state.position
    support_velocity = state.velocity + jnp.cross(state.angular_velocity, support_lever)
    material_state = state.material_state
    finger_normal = support_load = None
    if material is not None:
        if material_state is None:
            raise ValueError("Initialize material state before integrating a deformable object")
        all_points = jnp.concatenate((contact.reshape(-1, 3), support_points))
        local_points = (all_points - state.position) @ state.rotation
        indices, weights = surface_stencil(local_points, half_length, radius)
        count = penetration.size
        overlaps = jnp.concatenate((penetration.reshape(-1), support_penetration))
        vn = jnp.concatenate(
            (
                jnp.sum(relative * normal, axis=-1).reshape(-1),
                jnp.sum(support_velocity * support_normal, axis=-1),
            )
        )
        area = jnp.concatenate((jnp.full(count, 24e-6), jnp.full(support_penetration.size, 200e-6)))
        kp = jnp.concatenate((jnp.full(count, 4000.0), jnp.full(support_penetration.size, 12000.0)))
        cp = jnp.concatenate((jnp.full(count, 3.0), jnp.full(support_penetration.size, 60.0)))
        loads, material_state = material_response(
            material_state, material, indices, overlaps, vn, area, kp, cp, dt, weights=weights
        )
        finger_normal = loads[:count].reshape(penetration.shape)
        support_load = loads[count:]
    force, tangent, fn, ft, speed, sliding = friction_force(
        normal,
        penetration,
        relative,
        state.tangent,
        dt,
        mu_static[:, None, None],
        mu_dynamic[:, None, None],
        normal_force=finger_normal,
    )
    tool_force = -jnp.sum(force, axis=(1, 2))
    tool_moment = -jnp.sum(jnp.cross(contact - tool_position[:, None, None, :], force), axis=(1, 2))
    opening_force = jnp.sum(
        -force * tool_rotation[:, None, None, :, 1] * sides[None, :, None, None] / 2, axis=(1, 2, 3)
    )
    drive_width = state.drive_width + jnp.clip(
        target_width - state.drive_width, -0.15 * dt, 0.15 * dt
    )
    motor = jnp.clip(
        2000.0 * (drive_width - state.width) - 8.0 * state.width_velocity,
        -force_limit,
        force_limit,
    )
    if force_ceiling is not None:
        overload = jnp.max(jnp.sum(fn, axis=2), axis=1) > force_ceiling
        # Fast unloading bypasses the slow position-command ramp, not contact forces.
        # Impacts may still exceed the envelope and must be recorded, never clipped.
        motor = jnp.where(overload, force_limit, motor)
        drive_width = jnp.where(overload, jnp.minimum(state.width + 0.002, 0.10), drive_width)
        if material_state is not None:
            at_risk = (jnp.max(material_state.risk) > 0.55) & protect_material
            # Remove compression drive and brake the jaws rather than kicking open.
            motor = jnp.where(
                at_risk & ~overload,
                jnp.clip(-8.0 * state.width_velocity, -force_limit, force_limit),
                motor,
            )
            drive_width = jnp.where(at_risk & ~overload, state.width, drive_width)
    width_velocity = state.width_velocity + dt * (motor + opening_force) / 0.05
    proposed = state.width + dt * width_velocity
    width = jnp.clip(proposed, 0.004, 0.10)
    width_velocity = jnp.where(proposed == width, width_velocity, 0.0)

    support_force, support_tangent, support_fn, *_ = friction_force(
        support_normal,
        support_penetration,
        support_velocity,
        state.support_tangent,
        dt,
        0.7,
        0.5,
        stiffness=12000.0,
        damping=60.0,
        normal_force=support_load,
    )
    total_force = jnp.sum(force, axis=(0, 1, 2)) + jnp.sum(support_force, axis=0)
    moment = jnp.sum(jnp.cross(lever, force), axis=(0, 1, 2)) + jnp.sum(
        jnp.cross(support_lever, support_force), axis=0
    )
    # Phenomenological rolling resistance for finite compliant support patches.
    # Acts only while loaded; unlike global angular drag it vanishes in flight.
    rolling = (
        state.angular_velocity
        - jnp.sum(state.angular_velocity * support_normal, axis=1)[:, None] * support_normal
    )
    rolling_moment = (
        -rolling_resistance
        * radius
        * support_fn[:, None]
        * rolling
        / jnp.maximum(jnp.linalg.norm(rolling, axis=1)[:, None], 0.02)
    )
    moment += jnp.sum(rolling_moment, axis=0)
    world_inertia = state.rotation @ inertia @ state.rotation.T
    velocity = state.velocity + dt * (gravity + total_force / mass)
    omega = state.angular_velocity + dt * jnp.linalg.solve(
        world_inertia,
        moment - jnp.cross(state.angular_velocity, world_inertia @ state.angular_velocity),
    )
    angle = jnp.linalg.norm(omega) * dt
    orientation = (
        rotation(omega / jnp.maximum(jnp.linalg.norm(omega), 1e-12), angle) @ state.rotation
    )
    # Keep SO(3) invariants under long float32 rollouts; no pose attachment.
    x = orientation[:, 0] / jnp.linalg.norm(orientation[:, 0])
    y = orientation[:, 1] - jnp.dot(x, orientation[:, 1]) * x
    y = y / jnp.linalg.norm(y)
    orientation = jnp.stack((x, y, jnp.cross(x, y)), axis=1)
    following = ContactState(
        state.position + dt * velocity,
        orientation,
        velocity,
        omega,
        width,
        width_velocity,
        tangent,
        support_tangent,
        drive_width,
        material_state,
    )
    report = ContactReport(
        jnp.sum(fn, axis=2),
        jnp.sum(ft, axis=2),
        jnp.max(jnp.where(fn > 0.01, speed, 0.0), axis=2),
        jnp.any(sliding, axis=2),
        tool_force,
        tool_moment,
        jnp.max(jnp.maximum(penetration, 0.0), axis=(1, 2)),
        jnp.sum(fn, axis=2),
        jnp.max(material_state.risk) if material_state is not None else jnp.asarray(0.0),
        jnp.sum(jnp.where(support_normal[:, 2] > 0.9, support_fn, 0.0).reshape(-1, 3), axis=1),
    )
    return following, report
