"""Reference scenes built entirely from native Python model definitions."""

import numpy as np

from .spatial import ModelBuilder


def rod_inertia(mass, length, radius=0.035):
    """Solid cylinder aligned with local x, about its center of mass."""
    axial = mass * radius**2 / 2
    transverse = mass * (3 * radius**2 + length**2) / 12
    return np.diag([axial, transverse, transverse])


def arm6(*, dt=0.002, contacts=False):
    """Generic six-axis research arm (not a hardware-specific calibrated model)."""
    builder = ModelBuilder(dt=dt)
    base = builder.add_link("pedestal", joint="fixed", mass=0.0, inertia=np.zeros((3, 3)))
    lengths = (0.08, 0.32, 0.28, 0.08, 0.09, 0.09)
    masses = (2.0, 2.5, 1.8, 0.6, 0.4, 0.25)
    axes = ((0, 0, 1), (0, 1, 0), (0, 1, 0), (1, 0, 0), (0, 1, 0), (1, 0, 0))
    homes = (0.2, -0.8, 1.2, 0.0, -0.4, 0.0)
    names = ("waist", "shoulder", "elbow", "wrist_roll", "wrist_pitch", "tool_roll")
    parent = base
    for i, (length, mass, axis, home, name) in enumerate(zip(lengths, masses, axes, homes, names)):
        origin = (0, 0, 0.45) if i == 0 else (lengths[i - 1], 0, 0)
        parent = builder.add_link(
            name,
            parent=parent,
            axis=axis,
            origin=origin,
            mass=mass,
            com=(length / 2, 0, 0),
            inertia=rod_inertia(mass, length),
            damping=0.12,
            effort=60.0 if i < 3 else 12.0,
            limits=(-2.8, 2.8),
            home=home,
        )
        if contacts:
            builder.add_sphere(parent, (length, 0, 0), 0.035)
    builder.add_link(
        "tool",
        parent=parent,
        joint="fixed",
        origin=(lengths[-1], 0, 0),
        mass=0.05,
        inertia=np.eye(3) * 1e-5,
    )
    return builder.build()


def slider(*, dt=0.001, height=0.6, friction=0.6):
    """A vertically guided 1 kg ball for contact/convergence experiments."""
    builder = ModelBuilder(dt=dt)
    body = builder.add_link(
        "ball",
        joint="prismatic",
        axis=(0, 0, 1),
        mass=1.0,
        inertia=np.eye(3) * 0.004,
        home=height,
        damping=0.0,
        effort=30.0,
    )
    builder.add_sphere(body, (0, 0, 0), 0.1)
    return builder.build(friction=friction)


def multi_arm_urdf(count=2):
    """Compose one to four six-axis arms on a common fixed world frame."""
    import copy
    import xml.etree.ElementTree as ET
    from importlib.resources import files

    if not isinstance(count, int) or not 1 <= count <= 4:
        raise ValueError("Reference rig supports one to four arms")
    template = ET.fromstring(files("robogogogo").joinpath("assets/arm6.urdf").read_text())
    robot = ET.Element("robot", name=f"RG-{count}x6 Research Rig")
    ET.SubElement(robot, "link", name="world")
    for i in range(count):
        prefix = ("left" if i == 0 else "right") if count == 2 else f"arm_{i + 1}"
        for child in template:
            element = copy.deepcopy(child)
            element.set("name", f"{prefix}_{element.attrib['name']}")
            if element.tag == "joint":
                for tag in ("parent", "child"):
                    link = element.find(tag)
                    link.set("link", f"{prefix}_{link.attrib['link']}")
            robot.append(element)
        joint = ET.SubElement(robot, "joint", name=f"{prefix}_mount", type="fixed")
        ET.SubElement(joint, "parent", link="world")
        ET.SubElement(joint, "child", link=f"{prefix}_pedestal")
        ET.SubElement(joint, "origin", xyz=f"0 {((count - 1) / 2 - i) * 0.8} 0")
    return ET.tostring(robot, encoding="unicode")


def with_camera_mast(model, visuals):
    """Add a massless fixed head-camera fixture to the demo (not a humanoid neck)."""
    from dataclasses import replace

    import jax.numpy as jnp

    body = len(model.names)
    values = dict(
        origin=[-0.12, 0, 1.25],
        orientation=np.eye(3),
        axis=[1, 0, 0],
        mass=0.0,
        com=[0, 0, 0],
        inertia=np.zeros((3, 3)),
    )
    arrays = {
        name: jnp.concatenate(
            (getattr(model, name), jnp.asarray(value, dtype=model.home.dtype)[None]), axis=0
        )
        for name, value in values.items()
    }
    model = replace(
        model,
        names=model.names + ("head",),
        parents=model.parents + (0,),
        kinds=model.kinds + ("fixed",),
        indices=model.indices + (-1,),
        **arrays,
    )
    visuals = list(visuals) + [
        dict(
            body=0,
            shape="box",
            dimensions=[0.035, 0.035, 1.25],
            origin=[-0.12, 0, 0.625],
            rpy=[0, 0, 0],
        ),
        dict(body=body, shape="box", dimensions=[0.1, 0.16, 0.07], origin=[0, 0, 0], rpy=[0, 0, 0]),
    ]
    return model, visuals


def with_pan_tilt_head(model, visuals):
    """Optional two-DOF camera head with native dynamics; arm groups stay unchanged."""
    from dataclasses import replace

    import jax.numpy as jnp

    model, visuals = with_camera_mast(model, visuals)
    pan, pitch, joint = len(model.names) - 1, len(model.names), len(model.home)
    arrays = {}
    values = dict(
        origin=[0, 0, 0],
        orientation=np.eye(3),
        axis=[0, 1, 0],
        mass=0.15,
        com=[0.02, 0, 0],
        inertia=np.eye(3) * 0.001,
    )
    for name, value in values.items():
        array = getattr(model, name)
        if name == "axis":
            array = array.at[pan].set(jnp.array([0.0, 0.0, 1.0]))
        elif name == "mass":
            array = array.at[pan].set(0.2)
        elif name == "inertia":
            array = array.at[pan].set(jnp.eye(3) * 0.002)
        arrays[name] = jnp.concatenate((array, jnp.asarray(value, dtype=model.home.dtype)[None]))
    for name, value in dict(
        home=[0, 0], lower=[-0.7, -0.45], upper=[0.7, 0.45], damping=[0.1, 0.1], effort=[5.0, 5.0]
    ).items():
        arrays[name] = jnp.concatenate(
            (getattr(model, name), jnp.asarray(value, dtype=model.home.dtype))
        )
    model = replace(
        model,
        names=model.names[:-1] + ("head_pan", "head"),
        parents=model.parents + (pan,),
        kinds=model.kinds[:-1] + ("revolute", "revolute"),
        indices=model.indices[:-1] + (joint, joint + 1),
        **arrays,
    )
    visuals = [dict(v, body=pitch) if v["body"] == pan else v for v in visuals]
    return model, visuals
