"""Dimensioned provisional shoulder packaging, shared by render and motion guard.

These are reserved component volumes, not vendor CAD. Envelopes deliberately
include the complete motor/reducer stack; shaft and bearing interfaces may share
structural volume. Cable passages reserve space only; articulated harness design
and connector selection remain unresolved.
"""

import numpy as np

from ..planning.collision import Capsule


def shoulder_packages(model):
    from .components import ARM_COMPONENTS, mounting_offset, mounting_sign

    result = []
    for side in ("left", "right"):
        for suffix in ("waist", "upper_roll", "shoulder", "elbow"):
            body = model.names.index(f"{side}_{suffix}")
            part = ARM_COMPONENTS[suffix]
            axis = (
                np.asarray(model.orientation[body])
                @ np.asarray(model.axis[body])
                * mounting_sign(side, suffix)
            )
            origin = np.asarray(model.origin[body])
            offset = mounting_offset(suffix)
            result.append(
                dict(
                    name=f"{side}_{suffix}_drive_package",
                    body=model.parents[body],
                    start=(origin + axis * offset).tolist(),
                    end=(origin + axis * (offset + part.length_m)).tolist(),
                    radius=part.diameter_m / 2,
                    component=part.to_dict(),
                    provenance="manufacturer external dimensions; custom mounting interfaces",
                )
            )
    return result


def packaging_capsules(model):
    # A sphere enclosing each complete finite cylinder. More conservative than
    # vendor CAD, but it cannot falsely clear intersecting module bodies.
    return [
        Capsule(
            p["name"],
            p["body"],
            tuple((np.array(p["start"]) + p["end"]) / 2),
            tuple((np.array(p["start"]) + p["end"]) / 2),
            float(np.hypot(p["radius"], np.linalg.norm(np.array(p["end"]) - p["start"]) / 2)),
        )
        for p in shoulder_packages(model)
    ]


def arm_cable_passages(model):
    # Straight center bores in rigid links; no invented flexible cable crossing a joint.
    return [
        dict(
            name=f"{side}_{link}_cable_passage",
            body=model.names.index(f"{side}_{link}"),
            start=[0.025, 0, 0],
            end=[length - 0.025, 0, 0],
            radius=0.018,
            bundle_radius=0.015,
            status="rigid loom passage; proximal external service loops modeled; distal joint harness unresolved",
        )
        for side in ("left", "right")
        for link, length in (("shoulder", 0.32), ("elbow", 0.28))
    ]
