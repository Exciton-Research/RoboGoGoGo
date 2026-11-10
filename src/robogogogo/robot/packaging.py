"""Dimensioned provisional shoulder packaging, shared by render and motion guard.

These are reserved component volumes, not vendor CAD. Envelopes deliberately
include the complete motor/reducer stack; shaft and bearing interfaces may share
structural volume. Cable passages reserve space only; articulated harness design
and connector selection remain unresolved.
"""

import numpy as np

from ..planning.collision import Capsule


def shoulder_packages(model):
    result = []
    for side in ("left", "right"):
        for suffix in ("waist", "upper_roll", "shoulder"):
            body = model.names.index(f"{side}_{suffix}")
            sign = -1 if side == "right" and suffix == "shoulder" else 1
            axis = np.asarray(model.orientation[body]) @ np.asarray(model.axis[body]) * sign
            origin = np.asarray(model.origin[body])
            result.append(
                dict(
                    name=f"{side}_{suffix}_drive_package",
                    body=model.parents[body],
                    start=(origin + axis * 0.05).tolist(),
                    end=(origin + axis * 0.08).tolist(),
                    radius=0.023,
                    motor_radius=0.020,
                    reducer_length=0.015,
                    provenance="assumed packaging reservation; no selected components",
                )
            )
    return result


def packaging_capsules(model):
    return [
        Capsule(p["name"], p["body"], tuple(p["start"]), tuple(p["end"]), p["radius"])
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
            radius=0.006,
            bundle_radius=0.003,
            status="reserved; joint-to-joint harness unresolved",
        )
        for side in ("left", "right")
        for link, length in (("shoulder", 0.32), ("elbow", 0.28))
    ]
