"""The single Panda task: official robot, workbench, and a free-joint block."""

import xml.etree.ElementTree as ET

import numpy as np

TABLE_Z = 0.74
HALF_CUBE = 0.025
TARGET = np.array([0.47, 0.19, TABLE_Z + HALF_CUBE])
HOME = np.array([0, -0.35, 0, -2.1, 0, 1.75, -0.7854])


def scene_xml(path, *, object_spec=None, object_dir=None):
    root = ET.parse(path).getroot()
    root.find("compiler").set("meshdir", str(path.parent / "assets"))
    root.find("option").set("timestep", "0.002")
    root.find("option").set("iterations", "60")
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
        # 300 N/m tendon gain supports household-object loads within the
        # upstream 100 N force limit; contact still supplies all grasp forces.
        gripper = root.find("actuator/general[@name='actuator8']")
        gripper.set("gainprm", "0.04705882353 0 0")
        gripper.set("biasprm", "0 -300 -30")
        cube.remove(cube.find("geom"))
        asset = root.find("asset")
        ET.SubElement(asset, "mesh", name="object_mesh", file=str(object_dir / "visual.obj"))
        ET.SubElement(
            asset, "texture", name="object_texture", type="2d", file=str(object_dir / "texture.png")
        )
        ET.SubElement(
            asset,
            "material",
            name="object_material",
            texture="object_texture",
            specular="0.15",
            shininess="0.1",
        )
        ET.SubElement(
            cube,
            "geom",
            name="object_visual",
            type="mesh",
            mesh="object_mesh",
            material="object_material",
            contype="0",
            conaffinity="0",
            mass="0",
            group="2",
        )
        half = np.asarray(object_spec["dimensions"]) / 2
        size = [max(half[:2]), half[2]] if object_spec["collision"] == "cylinder" else half
        ET.SubElement(
            cube,
            "geom",
            name="object_collision",
            type=object_spec["collision"],
            size=" ".join(map(str, size)),
            mass=str(object_spec["mass"]),
            friction=f"{object_spec['friction']} 0.01 0.0001",
            condim="4",
            solref="0.005 1",
            group="3",
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
