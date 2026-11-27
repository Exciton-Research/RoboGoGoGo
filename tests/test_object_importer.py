"""Import actual textured geometry without downloading assets or model weights."""

import io
import json
import struct
import zipfile

import numpy as np
import pytest
from PIL import Image

from robogogogo.demo.importer import commit_import, prepare_import
from robogogogo.demo.objects import CATALOG, catalog, clutter_objects, require_object

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("mujoco")


def textured_box():
    mesh = trimesh.creation.box(extents=[1, 2, 3])
    image = Image.new("RGB", (8, 8), (220, 30, 50))
    image.paste((20, 90, 230), (0, 0, 4, 8))
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=(mesh.vertices[:, :2] + [0.5, 1]) / [1, 2], image=image
    )
    return trimesh.Scene(mesh)


@pytest.mark.parametrize("kind", ["glb", "zip"])
def test_textured_import_compiles_and_persists_outside_catalog(tmp_path, kind):
    scene = textured_box()
    if kind == "glb":
        payload = scene.export(file_type="glb")
    else:
        mesh = next(iter(scene.geometry.values()))
        obj, resources = trimesh.exchange.obj.export_obj(mesh, return_texture=True)
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("object.obj", obj)
            for name, data in resources.items():
                archive.writestr(name, data)
        payload = stream.getvalue()
    result = prepare_import(tmp_path, payload, f"object.{kind}", name="Test package", up="Z")
    spec = result["object"]
    assert spec["id"] not in CATALOG
    assert not spec["reference_matching"]
    np.testing.assert_allclose(spec["dimensions"], [0.04, 0.08, 0.12])
    assert len(catalog(tmp_path)) == 10  # Preview is not yet in the library.
    saved = commit_import(tmp_path, result["draft_id"])
    assert require_object(tmp_path, saved["id"])["sha256"] == saved["sha256"]
    assert len(catalog(tmp_path)) == 11
    assert len(clutter_objects(saved["id"])) == 9
    texture = Image.open(tmp_path / "imports" / saved["id"] / "texture.png")
    assert len(texture.getcolors()) >= 2
    with pytest.raises(ValueError, match="expired"):
        commit_import(tmp_path, result["draft_id"])
    # Re-importing the same source repairs a damaged local asset cache.
    folder = tmp_path / "imports" / saved["id"]
    (folder / "texture.png").write_bytes(b"damaged")
    replacement = prepare_import(tmp_path, payload, f"object.{kind}", name="Test package", up="Z")
    commit_import(tmp_path, replacement["draft_id"])
    assert require_object(tmp_path, saved["id"])["ready"]


@pytest.mark.parametrize("path", ["../object.obj", "/object.obj", "dir\\object.obj"])
def test_zip_paths_cannot_escape_the_import(tmp_path, path):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(path, "v 0 0 0")
    with pytest.raises(ValueError, match="unsafe path"):
        prepare_import(tmp_path, stream.getvalue(), "object.zip", name="Bad file")
    assert not (tmp_path / "imports").exists()


def test_glb_external_resources_are_rejected_before_loading(tmp_path):
    document = json.dumps(
        {"asset": {"version": "2.0"}, "buffers": [{"uri": "file:///private/data", "byteLength": 4}]}
    ).encode()
    document += b" " * (-len(document) % 4)
    payload = (
        struct.pack("<4sIII4s", b"glTF", 2, 20 + len(document), len(document), b"JSON") + document
    )
    with pytest.raises(ValueError, match="external references"):
        prepare_import(tmp_path, payload, "object.glb", name="Bad file")


def test_turn_is_applied_after_alignment_and_invalid_scale_is_rejected(tmp_path):
    payload = textured_box().export(file_type="glb")
    result = prepare_import(tmp_path, payload, "box.glb", name="Box", up="Z", turn=90)
    np.testing.assert_allclose(result["object"]["dimensions"], [0.08, 0.04, 0.12])
    assert not result["collectable"]
    with pytest.raises(ValueError, match="size"):
        prepare_import(tmp_path, payload, "box.glb", name="Box", size_cm=float("nan"))
