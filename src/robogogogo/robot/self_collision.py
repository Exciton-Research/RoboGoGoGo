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
        add(f"{side}_upper_arm", f"{side}_shoulder", (0.075, 0, 0), (0.245, 0, 0), 0.075)
        add(f"{side}_forearm", f"{side}_elbow", (0.065, 0, 0), (0.215, 0, 0), 0.065)
        add(f"{side}_elbow_cap", f"{side}_elbow", (0, -0.07, 0), (0, 0.07, 0), 0.057)
        add(f"{side}_wrist_roll", f"{side}_wrist_roll", (0, 0, 0), (0.04, 0, 0), 0.044)
        add(f"{side}_wrist_pitch", f"{side}_wrist_pitch", (0, 0, 0), (0.03, 0, 0), 0.044)
        add(f"{side}_wrist_tip", f"{side}_tool_roll", (0, 0, 0), (0.04, 0, 0), 0.041)
        add(f"{side}_hand", f"{side}_tool", (0.015, 0, 0), (0.12, 0, 0), 0.045)
        for name in ("chest_lower", "chest_middle", "chest_upper"):
            exclusions.append((f"{side}_socket", name))
        attached = [
            ("socket", "upper_arm"),
            ("upper_arm", "forearm"),
            ("upper_arm", "elbow_cap"),
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
    from .packaging import packaging_capsules

    internals = packaging_capsules(model)
    # Each package is enclosed by its existing socket envelope. Comparing it
    # against that robot's outer structural envelopes would double-count designed
    # containment. Inter-package pairs remain active, including adjacent axes.
    exclusions.extend((part.name, shell.name) for part in internals for shell in capsules)
    capsules.extend(internals)
    return CollisionWorld(model, capsules, margin=0.012, excluded_capsule_pairs=exclusions)
