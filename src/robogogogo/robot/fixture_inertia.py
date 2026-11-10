"""Aggregate fixed clamp masses into host links without introducing coordinates."""

import numpy as np


def add_fixture_inertia(robot, fixtures):
    for fixture in fixtures:
        element = robot.find(f"link[@name='{fixture['body_name']}']/inertial")
        mass = element.find("mass")
        origin = element.find("origin")
        inertia = element.find("inertia")
        old_mass, added_mass = float(mass.get("value")), fixture["mass_kg"]
        old_center = np.fromstring(origin.get("xyz"), sep=" ")
        a, b = np.array(fixture["start"]), np.array(fixture["end"])
        center = (a + b) / 2
        length = np.linalg.norm(b - a)
        axis = (b - a) / length
        radius = fixture["radius"]
        axial = added_mass * radius**2 / 2
        radial = added_mass * (3 * radius**2 + length**2) / 12
        added_tensor = np.eye(3) * radial + np.outer(axis, axis) * (axial - radial)
        tags = (("ixx", "ixy", "ixz"), ("ixy", "iyy", "iyz"), ("ixz", "iyz", "izz"))
        tensor = np.array([[float(inertia.get(tag)) for tag in row] for row in tags])
        combined = (old_center * old_mass + center * added_mass) / (old_mass + added_mass)
        for weight, point in ((old_mass, old_center), (added_mass, center)):
            displacement = point - combined
            tensor += weight * (
                np.eye(3) * np.dot(displacement, displacement)
                - np.outer(displacement, displacement)
            )
        tensor += added_tensor
        mass.set("value", str(old_mass + added_mass))
        origin.set("xyz", " ".join(map(str, combined)))
        for i, row in enumerate(tags):
            for j, tag in enumerate(row):
                inertia.set(tag, str(tensor[i, j]))
