"""Fetch the official, pinned Panda model into a local asset cache."""

import hashlib
import json
import ssl
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REVISION = "fb36886b3023db08c198264e1e9956ab202c3e3b"
SOURCE = (
    "https://github.com/google-deepmind/mujoco_menagerie/tree/" + REVISION + "/franka_emika_panda"
)
RAW = (
    "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/"
    + REVISION
    + "/franka_emika_panda/"
)


def model_path(cache):
    return Path(cache) / REVISION / "panda.xml"


def download(cache):
    """Explicit setup step. No asset downloads happen on an HTTP request."""
    import certifi

    directory = model_path(cache).parent
    directory.mkdir(parents=True, exist_ok=True)
    context = ssl.create_default_context(cafile=certifi.where())
    manifest = directory / "source.json"
    expected = json.loads(manifest.read_text()).get("sha256", {}) if manifest.is_file() else {}

    def fetch(name):
        path = directory / name
        intact = path.is_file() and (
            name not in expected or hashlib.sha256(path.read_bytes()).hexdigest() == expected[name]
        )
        if not intact:
            request = urllib.request.Request(RAW + name, headers={"User-Agent": "robogogogo-demo"})
            with urllib.request.urlopen(request, timeout=60, context=context) as response:
                payload = response.read()
            path.parent.mkdir(parents=True, exist_ok=True)
            pending = path.with_suffix(path.suffix + ".download")
            pending.write_bytes(payload)
            pending.replace(path)
        return name, hashlib.sha256(path.read_bytes()).hexdigest()

    fetch("panda.xml")
    root = ET.parse(directory / "panda.xml").getroot()
    files = ["panda.xml", "LICENSE", "README.md"]
    for mesh in root.findall(".//mesh"):
        name = mesh.get("file")
        if not name or Path(name).name != name:
            raise ValueError("Unexpected path in upstream Panda model")
        files.append("assets/" + name)
    files = sorted(set(files))
    print(f"Preparing official Panda model ({len(files)} files)…", flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        digests = dict(pool.map(fetch, files))
    (directory / "source.json").write_text(
        json.dumps(
            {
                "source": SOURCE,
                "revision": REVISION,
                "sha256": digests,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Panda assets ready: {directory}", flush=True)
    return directory / "panda.xml"


def require_model(cache):
    path = model_path(cache)
    manifest = path.parent / "source.json"
    if not manifest.is_file():
        raise FileNotFoundError("Panda assets are missing. Run robogogogo-demo --setup first.")
    source = json.loads(manifest.read_text())
    if source.get("revision") != REVISION:
        raise ValueError("Panda asset revision does not match the demo")
    for name, expected in source["sha256"].items():
        asset = path.parent / name
        if not asset.is_file() or hashlib.sha256(asset.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Panda asset changed or is incomplete: {name}")
    return path
