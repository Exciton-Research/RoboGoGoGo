"""Local, self-contained rigid-object imports. No model data or catalog matching."""

import base64
import hashlib
import io
import json
import shutil
import stat
import struct
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath
from uuid import UUID, uuid4

import numpy as np
from PIL import Image

MAX_UPLOAD = 32 * 1024 * 1024
MAX_UNPACKED = 96 * 1024 * 1024
MAX_FACES = 100_000
MAX_VERTICES = 200_000


def is_import(identifier):
    return (
        isinstance(identifier, str)
        and identifier.startswith("import-")
        and len(identifier) == 39
        and all(c in "0123456789abcdef" for c in identifier[7:])
    )


def import_directory(cache, identifier):
    if not is_import(identifier):
        raise ValueError("Invalid imported-object identifier")
    return Path(cache) / "imports" / identifier


def draft_directory(cache, identifier):
    return Path(cache) / "imports" / ".drafts" / str(UUID(str(identifier)))


def _image(data):
    with Image.open(io.BytesIO(data)) as image:
        if image.width * image.height > 16_777_216:
            raise ValueError("Textures must contain at most 16 million pixels")
        image.verify()


def _glb(payload):
    if len(payload) < 20 or struct.unpack_from("<4sII", payload) != (b"glTF", 2, len(payload)):
        raise ValueError("Choose a valid, self-contained GLB 2.0 file")
    length, kind = struct.unpack_from("<II", payload, 12)
    if kind != 0x4E4F534A or 20 + length > len(payload):
        raise ValueError("GLB has an invalid JSON chunk")
    document = json.loads(payload[20 : 20 + length])
    if not isinstance(document, dict):
        raise ValueError("GLB JSON must contain an object")
    if document.get("skins") or document.get("animations"):
        raise ValueError("Export a static rigid object without animation or skinning")
    if {"KHR_draco_mesh_compression", "EXT_meshopt_compression"} & set(
        document.get("extensionsUsed", [])
    ):
        raise ValueError("Export an uncompressed GLB")
    if any(m.get("alphaMode", "OPAQUE") != "OPAQUE" for m in document.get("materials", [])):
        raise ValueError("Export opaque materials; transparent objects are not supported")
    binary = payload[28 + length :] if len(payload) >= 28 + length else b""
    buffers = document.get("buffers", [])
    for resource in [*buffers, *document.get("images", [])]:
        uri = resource.get("uri")
        if uri is not None and not uri.startswith("data:"):
            raise ValueError(
                "GLB must embed its buffers and textures; external references are not loaded"
            )
    for image in document.get("images", []):
        if "uri" in image:
            data = base64.b64decode(image["uri"].split(",", 1)[1], validate=True)
        else:
            view = document["bufferViews"][image["bufferView"]]
            if view.get("buffer", 0) != 0:
                raise ValueError("Use embedded GLB image buffers")
            start = view.get("byteOffset", 0)
            data = binary[start : start + view["byteLength"]]
        _image(data)
    for accessor in document.get("accessors", []):
        if accessor.get("count", 0) > 3 * MAX_FACES:
            raise ValueError("Reduce the mesh to at most 100,000 triangles")
    return document


def _load(payload, filename):
    import trimesh

    suffix = Path(filename).suffix.lower()
    if suffix == ".glb":
        _glb(payload)
        return trimesh.load_scene(
            io.BytesIO(payload), file_type="glb", resolver={}, allow_remote=False, process=False
        ), "Y"
    if suffix != ".zip":
        raise ValueError("Choose a GLB, or a ZIP containing one OBJ with its MTL and textures")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        entries = archive.infolist()
        if len(entries) > 256 or sum(item.file_size for item in entries) > MAX_UNPACKED:
            raise ValueError("ZIP exceeds 256 files or 96 MB unpacked")
        assets = {}
        for item in entries:
            path = PurePosixPath(item.filename)
            mode = item.external_attr >> 16
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in item.filename
                or ":" in item.filename
                or stat.S_ISLNK(mode)
            ):
                raise ValueError("ZIP contains an unsafe path or symbolic link")
            if item.is_dir() or "__MACOSX" in path.parts or path.name.startswith("."):
                continue
            if path.suffix.lower() not in (".obj", ".mtl", ".png", ".jpg", ".jpeg", ".webp"):
                continue
            if str(path) in assets:
                raise ValueError("ZIP contains duplicate asset paths")
            data = archive.read(item)
            if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                _image(data)
            assets[str(path)] = data
    objects = [key for key in assets if key.lower().endswith(".obj")]
    if len(objects) != 1:
        raise ValueError("ZIP must contain exactly one OBJ")
    obj = objects[0]
    raw = assets[obj].decode("utf-8-sig")
    vertices = sum(line.startswith("v ") for line in raw.splitlines())
    triangles = sum(
        max(0, len(line.split()) - 3) for line in raw.splitlines() if line.startswith("f ")
    )
    if vertices > MAX_VERTICES or triangles > MAX_FACES:
        raise ValueError("Reduce the mesh to at most 200,000 vertices and 100,000 triangles")
    resolver = trimesh.resolvers.ZipResolver(assets, namespace=str(PurePosixPath(obj).parent))
    # Missing resources must not silently turn a textured object gray.
    for line in raw.splitlines():
        if line.startswith("mtllib "):
            try:
                mtl = resolver.get(line[7:].strip()).decode("utf-8-sig")
                for material_line in mtl.splitlines():
                    if material_line.strip().startswith("map_Kd "):
                        import shlex

                        resolver.get(shlex.split(material_line.strip())[1:][-1])
            except (KeyError, UnicodeError) as error:
                raise ValueError(
                    "Include every referenced MTL and base-color texture in the ZIP"
                ) from error
    return trimesh.load_scene(
        io.StringIO(raw),
        file_type="obj",
        resolver=resolver,
        allow_remote=False,
        process=False,
        group_material=True,
    ), "Z"


def _pack(meshes):
    import trimesh
    from trimesh.visual.material import SimpleMaterial, pack

    vertices, faces, normals, materials, uvs = [], [], [], [], []
    offset = 0
    for mesh in meshes:
        if mesh.visual.kind == "texture":
            uv, material = mesh.visual.uv, mesh.visual.material
            if uv is None or len(uv) != len(mesh.vertices):
                raise ValueError("Textured meshes need valid UV coordinates")
            if isinstance(material, trimesh.visual.material.MultiMaterial):
                raise ValueError("Export GLB with separate material primitives")
        else:
            visual = mesh.visual.to_texture()
            uv, material = visual.uv, visual.material
            if material is None:
                material = SimpleMaterial(diffuse=[200, 200, 200, 255])
        if not np.isfinite(uv).all():
            raise ValueError("Mesh UV coordinates must be finite")
        vertices.append(mesh.vertices)
        faces.append(mesh.faces + offset)
        normals.append(mesh.vertex_normals)
        materials.append(material)
        uvs.append(uv)
        offset += len(mesh.vertices)
    material, uv = pack(materials, uvs, max_tex_size_individual=2048, max_tex_size_fused=2048)
    texture = getattr(material, "image", None)
    if texture is None:
        texture = material.baseColorTexture
    return np.vstack(vertices), np.vstack(faces), np.vstack(normals), uv, texture


def _obj(path, vertices, faces, uv=None, normals=None):
    with path.open("w") as out:
        for point in vertices:
            out.write("v " + " ".join(f"{value:.9f}" for value in point) + "\n")
        if uv is not None:
            for point in uv:
                out.write("vt " + " ".join(f"{value:.9f}" for value in point) + "\n")
        if normals is not None:
            for point in normals:
                out.write("vn " + " ".join(f"{value:.9f}" for value in point) + "\n")
        for face in faces + 1:
            out.write(
                "f " + " ".join(f"{v}/{v}/{v}" if uv is not None else str(v) for v in face) + "\n"
            )


def _geometry(vertices, faces, uv=None, normals=None):
    def encode(values, kind):
        return base64.b64encode(np.asarray(values, dtype=kind).tobytes()).decode()

    result = {"vertices": encode(vertices, "<f4"), "faces": encode(faces, "<u4")}
    if uv is not None:
        result.update(uv=encode(uv, "<f4"), normals=encode(normals, "<f4"))
    return result


def prepare_import(
    cache, payload, filename, *, name, size_cm=12, up="auto", turn=0, mass_g=150, credit=""
):
    import mujoco
    import trimesh

    if not 0 < len(payload) <= MAX_UPLOAD:
        raise ValueError("Object files must be at most 32 MB")
    name, credit = name.strip(), credit.strip()
    if not 1 <= len(name) <= 120 or len(credit) > 300:
        raise ValueError("Name must contain 1–120 characters; credit is limited to 300")
    if not np.isfinite([size_cm, turn, mass_g]).all() or not (
        2 <= size_cm <= 22 and -180 <= turn <= 180 and 5 <= mass_g <= 500
    ):
        raise ValueError("Use a size of 2–22 cm, turn of -180–180 degrees, and mass of 5–500 g")
    scene, default_up = _load(payload, filename)
    if not scene.geometry or len(scene.graph.nodes_geometry) > 64:
        raise ValueError("Import one rigid object with at most 64 mesh parts")
    meshes = scene.dump()
    if any(not isinstance(mesh, trimesh.Trimesh) for mesh in meshes):
        raise ValueError("Only triangle meshes are supported")
    if (
        sum(len(m.vertices) for m in meshes) > MAX_VERTICES
        or sum(len(m.faces) for m in meshes) > MAX_FACES
    ):
        raise ValueError("Reduce the object to 200,000 vertices and 100,000 triangles")
    vertices, faces, normals, uv, texture = _pack(meshes)
    if not len(faces) or not np.isfinite(vertices).all():
        raise ValueError("Mesh must contain finite 3D geometry")
    up = default_up if up == "auto" else up
    if up not in ("X", "Y", "Z", "-X", "-Y", "-Z"):
        raise ValueError("Choose a valid up direction")
    axis = np.eye(3)["XYZ".index(up[-1])] * (-1 if up.startswith("-") else 1)
    rotation = trimesh.geometry.align_vectors(axis, [0, 0, 1])[:3, :3]
    yaw = trimesh.transformations.rotation_matrix(np.deg2rad(turn), [0, 0, 1])[:3, :3]
    vertices, normals = vertices @ rotation.T, normals @ rotation.T
    dimensions = np.ptp(vertices, axis=0)
    if dimensions.max() <= 0 or dimensions.min() / dimensions.max() < 0.005:
        raise ValueError("Object must have volume, not be a flat plane")
    scale = size_cm / 100 / dimensions.max()
    vertices = (vertices - (vertices.min(0) + vertices.max(0)) / 2) * scale
    # The importer, not a per-object grasp script, aligns the narrow side to the fingers.
    if np.ptp(vertices, axis=0)[0] > np.ptp(vertices, axis=0)[1]:
        align = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1])[:3, :3]
        vertices, normals = vertices @ align.T, normals @ align.T
    vertices, normals = vertices @ yaw.T, normals @ yaw.T
    dimensions = np.ptp(vertices, axis=0)
    hull = trimesh.Trimesh(vertices=vertices, faces=faces, process=False).convex_hull
    if not np.isfinite(hull.vertices).all() or hull.volume < 1e-10:
        raise ValueError("Could not create a solid collision hull")
    options = dict(name=name, size_cm=size_cm, up=up, turn=turn, mass_g=mass_g, credit=credit)
    digest = hashlib.sha256(payload).hexdigest()
    identifier = (
        "import-"
        + hashlib.sha256(
            (digest + json.dumps(options, sort_keys=True) + "import-v1").encode()
        ).hexdigest()[:32]
    )
    root = Path(cache) / "imports" / ".drafts"
    root.mkdir(parents=True, exist_ok=True)
    for old in root.iterdir():
        if old.is_dir() and old.stat().st_mtime < time.time() - 86400:
            shutil.rmtree(old)
    draft = str(uuid4())
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=root) as temporary:
        folder = Path(temporary)
        _obj(folder / "visual.obj", vertices, faces, uv, normals)
        _obj(folder / "collision.obj", hull.vertices, hull.faces)
        texture.convert("RGB").save(folder / "texture.png")
        spec = dict(
            id=identifier,
            name=name,
            category="Imported",
            imported=True,
            reference_matching=False,
            dimensions=dimensions.tolist(),
            mass=mass_g / 1000,
            collision="mesh",
            collision_file="collision.obj",
            friction=1.1,
            source=None,
            credit=credit,
            source_filename=Path(filename).name,
            source_sha256=digest,
            import_options=options,
            ready=True,
            physics="Rigid object with a convex collision proxy; assigned mass and friction",
            sha256={
                f: hashlib.sha256((folder / f).read_bytes()).hexdigest()
                for f in ("visual.obj", "texture.png", "collision.obj")
            },
        )
        (folder / "object.json").write_text(json.dumps(spec, indent=2) + "\n")
        # Compile the exact visual/collision assets before offering Load.
        import xml.etree.ElementTree as ET

        from .task import add_scan

        xml = ET.fromstring(
            '<mujoco><asset/><worldbody><body name="object"><freejoint/></body></worldbody></mujoco>'
        )
        add_scan(xml, xml.find("worldbody/body"), spec, folder)
        mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
        target = draft_directory(cache, draft)
        shutil.copytree(folder, target)
    return {
        "draft_id": draft,
        "object": spec,
        "visual": _geometry(vertices, faces, uv, normals),
        "collision": _geometry(hull.vertices, hull.faces),
        "texture_url": f"/api/demo/objects/import/drafts/{draft}/texture",
        "collectable": bool(dimensions[0] <= 0.075 and 0.015 <= dimensions[2] <= 0.22),
    }


def commit_import(cache, draft):
    folder = draft_directory(cache, draft)
    if not (folder / "object.json").is_file():
        raise ValueError("Import preview expired; preview the file again")
    spec = json.loads((folder / "object.json").read_text())
    target = import_directory(cache, spec["id"])
    if target.exists():
        from .objects import require_object

        try:
            existing = require_object(cache, spec["id"])
        except FileNotFoundError:
            # A freshly compiled preview can repair a damaged local cache entry.
            for name in ("visual.obj", "texture.png", "collision.obj", "object.json"):
                shutil.copyfile(folder / name, target / name)
            existing = require_object(cache, spec["id"])
        shutil.rmtree(folder)
        return existing
    folder.rename(target)
    return spec
