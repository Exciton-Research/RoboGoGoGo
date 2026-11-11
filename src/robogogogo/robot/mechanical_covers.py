"""Removable split motor sleeves for mechanical development.

Open ends preserve shaft and service access. These custom 2 mm shells reserve
1 mm radial clearance around catalog motor envelopes; fasteners and production
tolerance stacks are not qualified. The small split gap is ignored for inertia.
"""

import numpy as np

from .packaging import shoulder_packages


def motor_cover_specs(model):
    covers = []
    for package in shoulder_packages(model):
        a, b = np.array(package["start"]), np.array(package["end"])
        axis = (b - a) / np.linalg.norm(b - a)
        inner, outer = package["radius"] + 0.001, package["radius"] + 0.003
        length = np.linalg.norm(b - a) + 0.006
        mass = 1200 * np.pi * (outer**2 - inner**2) * length
        axial = mass * (outer**2 + inner**2) / 2
        radial = mass * (3 * (outer**2 + inner**2) + length**2) / 12
        covers.append(
            dict(
                name=package["name"].replace("drive_package", "split_cover"),
                body=package["body"],
                body_name=model.names[package["body"]],
                start=(a - axis * 0.003).tolist(),
                end=(b + axis * 0.003).tolist(),
                radius=outer,
                inner_radius=inner,
                seam_gap_m=0.0004,
                mass_kg=float(mass),
                local_inertia=(
                    np.eye(3) * radial + np.outer(axis, axis) * (axial - radial)
                ).tolist(),
            )
        )
    return covers
