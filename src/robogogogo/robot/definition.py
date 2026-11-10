"""Versioned identity for the only supported product embodiment."""

ROBOT_ID = "rg_mobile_duo"
ROBOT_REVISION = "engineering-1"
ARM_DOF = 7
SIMULATION_DOF = 31

# Public recording order uses the moving body names in the canonical model.
SIMULATION_JOINT_NAMES = (
    "base_x",
    "base_y",
    "base_link",
    *(
        f"{corner}_{joint}"
        for corner in ("front_left", "front_right", "rear_left", "rear_right")
        for joint in ("steer", "wheel")
    ),
    "torso_yaw",
    "torso_lower",
    "torso_upper",
    "torso",
    *(
        f"{side}_{joint}"
        for side in ("left", "right")
        for joint in (
            "waist",
            "upper_roll",
            "shoulder",
            "elbow",
            "wrist_roll",
            "wrist_pitch",
            "tool_roll",
        )
    ),
    "head_pan",
    "head",
)
