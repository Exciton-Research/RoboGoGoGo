"""Assumed output-side arm actuator envelope; not a selected motor specification.

Linear motoring torque-speed droop, full peak braking torque, and smooth Coulomb
friction. No thermal, bus-voltage, backlash, stiction or transmission compliance
model. All speeds are joint rad/s and torques are joint N m after gearing.
"""

import jax.numpy as jnp

ARM_ACTUATOR_REVISION = "catalog-arm-drive-2"
# suffix: no-load output speed, Coulomb friction torque
ARM_DRIVES = {
    "waist": (3.0, 0.12),
    "upper_roll": (3.0, 0.12),
    "shoulder": (3.0, 0.12),
    "elbow": (3.5, 0.08),
    "wrist_roll": (5.0, 0.025),
    "wrist_pitch": (5.0, 0.025),
    "tool_roll": (5.0, 0.025),
}


def arm_drive_parameters(model):
    speed = jnp.full_like(model.home, jnp.inf)
    friction = jnp.zeros_like(model.home)
    for side in ("left", "right"):
        for suffix, (no_load, coulomb) in ARM_DRIVES.items():
            index = model.indices[model.names.index(f"{side}_{suffix}")]
            if suffix in ARM_DRIVES:
                from .components import ARM_COMPONENTS

                no_load = ARM_COMPONENTS[suffix].no_load_rpm * 2 * jnp.pi / 60
            speed = speed.at[index].set(no_load)
            friction = friction.at[index].set(coulomb)
    return speed, friction


def delivered_torque(requested, velocity, effort, no_load_speed):
    """Limit motor effort without clamping physical joint velocity or position."""
    motoring_speed = jnp.maximum(jnp.sign(requested) * velocity, 0)
    available = effort * jnp.clip(1 - motoring_speed / no_load_speed, 0, 1)
    return jnp.clip(requested, -available, available)


def gearbox_friction(velocity, coulomb):
    """Passive, dissipative approximation; deliberately zero at rest (no stiction)."""
    return -coulomb * jnp.tanh(velocity / 0.02)


def arm_drive_export(model):
    """NPZ-safe provenance and parameters; infinity means no speed derating."""
    import numpy as np

    speed, friction = arm_drive_parameters(model)
    return {
        "arm_actuator_revision": np.array(ARM_ACTUATOR_REVISION),
        "arm_actuator_provenance": np.array(
            "arm catalog speed/continuous torque; friction and structural inertia assumed; not measured"
        ),
        "arm_no_load_speed_rad_s": np.asarray(speed),
        "arm_coulomb_friction_nm": np.asarray(friction),
        "arm_friction_smoothing_rad_s": np.array(0.02),
    }
