"""Two curated YCB scans. Downloads and derived assets stay in the local cache."""

import hashlib
import io
import json
import ssl
import tarfile
import tempfile
import threading
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

SOURCE = "https://ycb-benchmarks.s3.amazonaws.com/index.html"
LICENSE = "https://creativecommons.org/licenses/by/4.0/"
CREDIT = "YCB Object and Model Set — Calli et al.; Google 16k scans"
CATALOG = {
    "soup-can": {
        "name": "Tomato soup can",
        "category": "Household",
        "ycb": "005_tomato_soup_can",
        "archive_sha256": "264f28559475e8e548b20420a7b6db944b4b27471039a73733d2c08a4ba2cc3d",
        "mass": 0.35,
        "collision": "cylinder",
        "yaw": 0.0,
    },
    "rubiks-cube": {
        "name": "Rubik’s Cube",
        "category": "Toy",
        "ycb": "077_rubiks_cube",
        "archive_sha256": "f5a44bde4605f402247e7f4b5e2ef93469dabfd07062cb2b67efdcdbe088587c",
        "mass": 0.09,
        "collision": "box",
        "yaw": 1.0995574287564276,
    },
}
BLOCK = {
    "id": "block",
    "name": "Original block",
    "category": "Primitive",
    "dimensions": [0.05, 0.05, 0.05],
    "mass": 0.08,
    "collision": "box",
    "friction": 1.1,
    "source": None,
    "ready": True,
}
_DOWNLOAD_LOCK = threading.Lock()


def directory(cache, identifier):
    if identifier not in CATALOG:
        raise ValueError("Unknown object in the library")
    return Path(cache) / "objects" / identifier


def require_object(cache, identifier):
    if identifier == "block":
        return dict(BLOCK)
    folder = directory(cache, identifier)
    try:
        manifest = json.loads((folder / "object.json").read_text())
        if manifest["archive_sha256"] != CATALOG[identifier]["archive_sha256"]:
            raise ValueError("Object source changed")
        for name in ("visual.obj", "texture.png"):
            if hashlib.sha256((folder / name).read_bytes()).hexdigest() != manifest["sha256"][name]:
                raise ValueError("Cached object asset changed")
    except (OSError, KeyError, ValueError) as error:
        raise FileNotFoundError(
            f"Download {CATALOG[identifier]['name']} from the object library first"
        ) from error
    return {**manifest, "ready": True}


def catalog(cache):
    result = [dict(BLOCK)]
    for identifier, item in CATALOG.items():
        try:
            result.append(require_object(cache, identifier))
        except FileNotFoundError:
            result.append(
                {
                    "id": identifier,
                    "name": item["name"],
                    "category": item["category"],
                    "ready": False,
                    "source": SOURCE,
                    "license": LICENSE,
                }
            )
    return result


def download_object(cache, identifier):
    """Explicit setup or user action only; never called by scene/state reads."""
    import certifi

    if identifier == "block":
        return dict(BLOCK)
    folder = directory(cache, identifier)
    item = CATALOG[identifier]
    with _DOWNLOAD_LOCK:
        try:
            return require_object(cache, identifier)
        except FileNotFoundError:
            pass
        url = f"https://ycb-benchmarks.s3.amazonaws.com/data/google/{item['ycb']}_google_16k.tgz"
        context = ssl.create_default_context(cafile=certifi.where())
        request = urllib.request.Request(url, headers={"User-Agent": "robogogogo-object-library"})
        with urllib.request.urlopen(request, context=context, timeout=60) as response:
            payload = response.read(32 * 1024 * 1024 + 1)
        if hashlib.sha256(payload).hexdigest() != item["archive_sha256"]:
            raise ValueError("Object download did not match the pinned YCB archive")
        folder.mkdir(parents=True, exist_ok=True)
        # Read only named regular files; archive paths are never extracted.
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:

            def read(name, limit):
                member = archive.getmember(f"{item['ycb']}/google_16k/{name}")
                if not member.isfile() or member.size > limit:
                    raise ValueError("Unexpected object archive member")
                return archive.extractfile(member).read()

            raw = read("textured.obj", 4_000_000).decode("utf-8")
            image = read("texture_map.png", 16_000_000)
        lines = raw.splitlines()
        vertices = np.asarray(
            [
                [float(value) for value in line.split()[1:4]]
                for line in lines
                if line.startswith("v ")
            ]
        )
        angle = item["yaw"]
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        vertices = vertices @ rotation.T
        center = (vertices.min(0) + vertices.max(0)) / 2
        dimensions = np.ptp(vertices, axis=0)
        vertices -= center
        transformed = iter(vertices)
        output = []
        for line in lines:
            if line.startswith("v "):
                output.append("v " + " ".join(f"{value:.9f}" for value in next(transformed)))
            elif line.startswith("vn "):
                normal = rotation @ np.array([float(v) for v in line.split()[1:4]])
                output.append("vn " + " ".join(f"{v:.9f}" for v in normal))
            elif not line.startswith(("mtllib ", "usemtl ")):
                output.append(line)
        manifest = {
            "id": identifier,
            "name": item["name"],
            "category": item["category"],
            "dimensions": dimensions.tolist(),
            "mass": item["mass"],
            "collision": item["collision"],
            "friction": 1.1,
            "physics": "Assigned mass and friction; not calibrated physical measurements",
            "source": SOURCE,
            "download": url,
            "credit": CREDIT,
            "license": LICENSE,
            "ycb": item["ycb"],
            "archive_sha256": item["archive_sha256"],
            "processing": "Metric scale preserved; centered and rotated upright; texture reduced to 1024 px",
            "yaw": angle,
        }
        with tempfile.TemporaryDirectory(prefix=".prepare-", dir=folder.parent) as temporary:
            root = Path(temporary)
            (root / "visual.obj").write_text("\n".join(output) + "\n")
            with Image.open(io.BytesIO(image)) as texture:
                texture.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                texture.convert("RGB").save(root / "texture.png")
            manifest["sha256"] = {
                name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in ("visual.obj", "texture.png")
            }
            (root / "object.json").write_text(json.dumps(manifest, indent=2) + "\n")
            for name in ("visual.obj", "texture.png", "object.json"):
                (root / name).replace(folder / name)
        return {**manifest, "ready": True}
