"""Panda tabletop scenes with textured free objects and collection cells."""

import xml.etree.ElementTree as ET

import numpy as np

TABLE_Z = 0.74
HALF_CUBE = 0.025
TARGET = np.array([0.47, 0.19, TABLE_Z + HALF_CUBE])
HOME = np.array([0, -0.35, 0, -2.1, 0, 1.75, -0.7854])


def collection_xy(index, count):
    """Keep collection geometry and environment destinations in agreement."""
    if count > 8:
        return np.array([0.28 + (index % 3) * 0.19, 0.10 + (index // 3) * 0.13])
    return np.array([0.28 + (index % 4) * 0.15, 0.16 + (index // 4) * 0.14])


def scene_xml(
    path,
    *,
    object_spec=None,
    object_dir=None,
    other_objects=None,
    gripper_stiffness=600,
    collection_style="cells",
    soft_fingers=True,
):
    root = ET.parse(path).getroot()
    root.find("compiler").set("meshdir", str(path.parent / "assets"))
    root.find("option").set("timestep", "0.002")
    root.find("option").set("iterations", "60")
    if soft_fingers and object_spec is not None and object_spec["id"] != "block":
        # MuJoCo's recommended manipulation settings stiffen tangential
        # constraints without inflating the sliding-friction coefficient.
        root.find("option").set("cone", "elliptic")
        root.find("option").set("impratio", "10")
    root.remove(root.find("keyframe"))
    world = root.find("worldbody")
    world.find("body").set("pos", f"0 0 {TABLE_Z}")
    hand = world.find(".//body[@name='hand']")
    ET.SubElement(hand, "site", name="grasp_center", pos="0 0 0.1034", size="0.004")
    ET.SubElement(
        world, "geom", name="floor", type="plane", size="5 5 0.1", rgba="0.88 0.87 0.84 1"
    )
    ET.SubElement(
        world,
        "geom",
        name="worktop",
        type="box",
        size="0.66 0.48 0.035",
        pos=f"0.3 0 {TABLE_Z - 0.035}",
        rgba="0.72 0.74 0.72 1",
        friction="0.8 0.005 0.0001",
    )
    for x in (-0.25, 0.85):
        for y in (-0.37, 0.37):
            ET.SubElement(
                world,
                "geom",
                type="box",
                size="0.025 0.025 0.335",
                pos=f"{x} {y} 0.335",
                rgba="0.22 0.25 0.25 1",
            )
    cube = ET.SubElement(
        world, "body", name="block", pos=f"0.47 -0.17 {TABLE_Z + HALF_CUBE + 0.002}"
    )
    ET.SubElement(cube, "freejoint", name="block_free")
    ET.SubElement(
        cube,
        "geom",
        name="block_geom",
        type="box",
        size="0.025 0.025 0.025",
        mass="0.08",
        rgba="0.88 0.31 0.13 1",
        friction="1.1 0.01 0.0001",
        condim="4",
        solref="0.005 1",
    )
    if object_spec is not None and object_spec["id"] != "block":
        # The original light block uses a very soft gripper servo. A modest
        # 600 N/m tendon gain keeps household loads from sliding out of the
        # pads, within the upstream 100 N force limit. Contact supplies all forces.
        gripper = root.find("actuator/general[@name='actuator8']")
        gripper.set("gainprm", f"{0.04 * gripper_stiffness / 255:.11f} 0 0")
        gripper.set("biasprm", f"0 {-gripper_stiffness:g} {-gripper_stiffness * 0.1:g}")
        cube.remove(cube.find("geom"))
        add_scan(root, cube, object_spec, object_dir, rolling=soft_fingers)
    for index, (spec, folder) in enumerate(other_objects or [], start=1):
        body = ET.SubElement(
            world,
            "body",
            name=f"clutter_{index}",
            pos=f"0.47 -0.17 {TABLE_Z + spec['dimensions'][2] / 2 + 0.002}",
        )
        ET.SubElement(body, "freejoint", name=f"clutter_{index}_free")
        add_scan(root, body, spec, folder, suffix=f"_{index}", rolling=soft_fingers)
    if soft_fingers and object_spec is not None and object_spec["id"] != "block":
        # Soft pads resist rolling as well as sliding. Apply this only at finger
        # contacts. Other contacts retain their original friction coefficients,
        # with their already-declared rolling resistance enabled.
        contact = root.find("contact")
        if contact is None:
            contact = ET.SubElement(root, "contact")
        for side in ("left", "right"):
            finger = world.find(f".//body[@name='{side}_finger']")
            for index, geom in enumerate(finger.findall("geom")):
                if geom.get("class") == "visual":
                    continue
                name = f"{side}_contact_{index}"
                geom.set("name", name)
                for object_index, spec in enumerate(
                    [object_spec, *[item[0] for item in other_objects or []]]
                ):
                    sliding = max(1.0, float(spec["friction"]))
                    other = "object_collision" + (f"_{object_index}" if object_index else "")
                    ET.SubElement(
                        contact,
                        "pair",
                        geom1=name,
                        geom2=other,
                        condim="6",
                        friction=f"{sliding:g} {sliding:g} 0.01 0.001 0.001",
                        solref="0.005 1",
                    )
    if other_objects and collection_style == "bin":
        # A shared receiving bin gives long packages room to settle. Painted
        # target cells remain distinct; internal rails are unnecessary obstacles.
        for axis in range(2):
            for sign in (-1, 1):
                center, half = np.array([0.47, 0.2425]), np.array([0.285, 0.2075])
                position = [*center, TABLE_Z + 0.015]
                position[axis] += sign * half[axis]
                size = [*half, 0.015]
                size[axis] = 0.002
                ET.SubElement(
                    world,
                    "geom",
                    name=f"bin_{axis}_{sign}",
                    type="box",
                    pos=" ".join(map(str, position)),
                    size=" ".join(map(str, size)),
                    rgba="0.45 0.52 0.44 1",
                    friction="0.8 0.005 0.0001",
                )
        for index in range(1 + len(other_objects)):
            x, y = collection_xy(index, 1 + len(other_objects))
            ET.SubElement(
                world,
                "geom",
                name=f"cell_{index}",
                type="box",
                size="0.065 0.06 0.0002",
                pos=f"{x} {y} {TABLE_Z + 0.0002}",
                rgba="0.60 0.66 0.58 1",
                contype="0",
                conaffinity="0",
                group="2",
            )
    elif other_objects:
        for index in range(1 + len(other_objects)):
            x, y = collection_xy(index, 1 + len(other_objects))
            for axis in range(2):
                for sign in (-1, 1):
                    position = [x, y, TABLE_Z + 0.015]
                    position[axis] += sign * 0.065
                    size = [0.065, 0.065, 0.015]
                    size[axis] = 0.002
                    ET.SubElement(
                        world,
                        "geom",
                        name=f"collection_{index}_{axis}_{sign}",
                        type="box",
                        pos=" ".join(map(str, position)),
                        size=" ".join(map(str, size)),
                        rgba="0.45 0.52 0.44 1",
                        friction="0.8 0.005 0.0001",
                    )
    ET.SubElement(world, "camera", name="overhead", pos="0.47 0 1.75", fovy="42")
    ET.SubElement(
        hand,
        "camera",
        name="wrist",
        pos="0.06 0 0.045",
        xyaxes="-0.9734 0 -0.2290 0 1 0",
        fovy="65",
    )
    ET.SubElement(
        world,
        "light",
        pos="1.5 -1.5 4",
        dir="-1 1 -3",
        diffuse="0.7 0.7 0.7",
        ambient="0.15 0.15 0.15",
    )
    ET.SubElement(
        world,
        "geom",
        name="target_marker",
        type="box",
        size="0.065 0.065 0.0003",
        pos="0.47 0.19 0.7404",
        rgba="0.45 0.65 0.50 1",
        contype="0",
        conaffinity="0",
        group="2",
    )
    return ET.tostring(root, encoding="unicode")


def add_scan(root, body, spec, folder, suffix="", *, rolling=False):
    asset = root.find("asset")
    mesh, texture, material = (
        name + suffix for name in ("object_mesh", "object_texture", "object_material")
    )
    ET.SubElement(asset, "mesh", name=mesh, file=str(folder / "visual.obj"))
    ET.SubElement(asset, "texture", name=texture, type="2d", file=str(folder / "texture.png"))
    ET.SubElement(
        asset, "material", name=material, texture=texture, specular="0.15", shininess="0.1"
    )
    ET.SubElement(
        body,
        "geom",
        name="object_visual" + suffix,
        type="mesh",
        mesh=mesh,
        material=material,
        contype="0",
        conaffinity="0",
        mass="0",
        group="2",
    )
    half = np.asarray(spec["dimensions"]) / 2
    attributes = (
        {"mesh": mesh}
        if spec["collision"] == "mesh"
        else {
            "size": " ".join(
                map(str, [max(half[:2]), half[2]] if spec["collision"] == "cylinder" else half)
            )
        }
    )
    ET.SubElement(
        body,
        "geom",
        name="object_collision" + suffix,
        type=spec["collision"],
        **attributes,
        mass=str(spec["mass"]),
        friction=f"{spec['friction']} 0.01 0.0001",
        condim="6" if rolling else "4",
        solref="0.005 1",
        group="3",
    )
