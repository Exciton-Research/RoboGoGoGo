"""Explicit mobile robot collision envelopes and structural exemptions.

Capsules conservatively cover arm sleeves, wrist cuffs, gripper fingers and body
shells. Attached joints intentionally share volumes; exclusions are named here,
not inferred by disabling collisions for an entire arm or the shared torso.
"""

from itertools import combinations

from ..planning.collision import Capsule, CollisionWorld


def mobile_collision_world(model):
    capsules = []

    def add(name, link, start, end, radius):
        capsules.append(Capsule(name, model.names.index(link), start, end, radius))

    for side, y in (("left", 0.27), ("right", -0.27)):
        add(f"base_{side}", "base_link", (-0.28, y, 0), (0.28, y, 0), 0.15)
    add("base_rear", "base_link", (-0.27, -0.26, 0.02), (-0.27, 0.26, 0.02), 0.145)
    add("pelvis", "torso_yaw", (0, 0, 0), (0, 0, 0), 0.13)
    add("lower_torso", "torso_lower", (0, 0, 0.075), (0, 0, 0.365), 0.09)
    add("upper_torso", "torso_upper", (0, 0, 0.065), (0, 0, 0.375), 0.08)
    for name, z in (("chest_lower", 0.10), ("chest_middle", 0.14), ("chest_upper", 0.18)):
        add(name, "torso", (0, -0.11, z), (0, 0.11, z), 0.151)
    add("head", "head", (0, 0, 0), (0, 0, 0), 0.14)
    body_names = [c.name for c in capsules]
    # This guard addresses arm/body and arm/arm motion. The fixed design's body
    # assembly shares volumes at torso hinges; body/body validation is separate.
    exclusions = list(combinations(body_names, 2))
    for side in ("left", "right"):
        add(f"{side}_socket", f"{side}_waist", (0, 0, 0), (0, 0, 0), 0.17)
        from .arm_structure import ARM_STRUCTURE

        for suffix, label in (("shoulder", "upper_arm"), ("elbow", "forearm")):
            spec = ARM_STRUCTURE[suffix]
            add(
                f"{side}_{label}",
                f"{side}_{suffix}",
                (spec["sleeve_start"], 0, 0),
                (spec["sleeve_end"], 0, 0),
                spec["radius"],
            )
        add(f"{side}_wrist_roll", f"{side}_wrist_roll", (0, 0, 0), (0.055, 0, 0), 0.044)
        add(f"{side}_wrist_pitch", f"{side}_wrist_pitch", (0, 0, 0), (0.10, 0, 0), 0.044)
        add(f"{side}_wrist_tip", f"{side}_tool_roll", (0, 0, 0), (0.13, 0, 0), 0.041)
        add(f"{side}_hand", f"{side}_tool", (0.015, 0, 0), (0.12, 0, 0), 0.045)
        for name in ("chest_lower", "chest_middle", "chest_upper"):
            exclusions.append((f"{side}_socket", name))
        attached = [
            ("socket", "upper_arm"),
            ("upper_arm", "forearm"),
            ("forearm", "wrist_roll"),
            ("forearm", "wrist_pitch"),
            ("wrist_roll", "wrist_pitch"),
            ("wrist_roll", "wrist_tip"),
            ("wrist_roll", "hand"),
            ("wrist_pitch", "wrist_tip"),
            ("wrist_pitch", "hand"),
            ("wrist_tip", "hand"),
        ]
        exclusions.extend((f"{side}_{a}", f"{side}_{b}") for a, b in attached)
    from .packaging import mechanical_assembly_capsules, packaging_capsules

    internals = packaging_capsules(model) + mechanical_assembly_capsules(model)
    # Exempt only named local mating/containing envelopes, never the opposite
    # arm or remote body parts. Internal parts still check other drive assemblies.
    containing = {
        "waist": ("socket",),
        "upper_roll": ("socket",),
        "shoulder": ("socket", "upper_arm"),
        "elbow": ("upper_arm", "forearm"),
        "wrist_roll": ("forearm", "wrist_roll", "wrist_pitch"),
        "wrist_pitch": ("wrist_roll", "wrist_pitch", "wrist_tip"),
        "tool_roll": ("wrist_pitch", "wrist_tip", "hand"),
    }
    for part in internals:
        side = part.name.split("_", 1)[0]
        suffix = next(key for key in containing if part.name.startswith(f"{side}_{key}_"))
        for shell in containing[suffix]:
            exclusions.append((part.name, f"{side}_{shell}"))
        if suffix in ("waist", "upper_roll", "shoulder"):
            for chest in ("chest_lower", "chest_middle", "chest_upper"):
                exclusions.append((part.name, chest))
    # These shafts meet their own actuator output bearings. This is a named
    # mating interface, not an exemption for adjacent motors or the other arm.
    for side in ("left", "right"):
        for suffix, shaft in (
            ("waist", "lift_output"),
            ("upper_roll", "swing_output"),
            ("shoulder", "roll_output"),
            ("elbow", "arm_output"),
            ("wrist_roll", "arm_output"),
            ("wrist_pitch", "arm_output"),
            ("tool_roll", "arm_output"),
            ("elbow", "arm_output_link"),
            ("wrist_roll", "arm_output_link"),
            ("wrist_pitch", "arm_output_link"),
        ):
            exclusions.append((f"{side}_{suffix}_mount_{shaft}", f"{side}_{suffix}_drive_package"))
    capsules.extend(internals)
    # Slim motor sleeves fit inside each drive's existing 4 mm package allowance.
    # No separate cable-service hoods or electrical attachment exemptions remain.
    return CollisionWorld(model, capsules, margin=0.012, excluded_capsule_pairs=exclusions)
