"""User-authored tabletop layouts, stored beside the local object library."""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np

from .objects import require_object

# Footprints must stay inside the calibrated pickup workspace, clear of the bin.
WORKSPACE = (0.21, 0.81, -0.39, 0.015)
GAP = 0.008


def _specs(cache, definition):
    entries = definition.get("objects", [])
    ids = [entry["id"] for entry in entries]
    if not 1 <= len(ids) <= 9 or len(set(ids)) != len(ids) or "block" in ids:
        raise ValueError("Choose one to nine distinct scanned or imported objects")
    if definition.get("target") not in ids:
        raise ValueError("Choose a target that is on this table")
    name = definition.get("name", "").strip()
    seed = definition.get("seed", 0)
    if not 1 <= len(name) <= 80 or type(seed) is not int or not 0 <= seed <= 1_000_000:
        raise ValueError("Use a table name of 1–80 characters and a seed of 0–1000000")
    return [require_object(cache, identifier) for identifier in ids]


def overlaps(a, b, half_a, half_b):
    return bool(np.all(np.abs(np.asarray(a) - b) < half_a + half_b + GAP - 1e-9))


def validate_layout(cache, definition):
    specs = _specs(cache, definition)
    positions = np.asarray([[entry["x"], entry["y"]] for entry in definition["objects"]], float)
    half = np.asarray([spec["dimensions"][:2] for spec in specs]) / 2
    if not np.isfinite(positions).all():
        raise ValueError("Object positions must be finite")
    xmin, xmax, ymin, ymax = WORKSPACE
    for index, spec in enumerate(specs):
        low, high = positions[index] - half[index], positions[index] + half[index]
        if np.any(low < [xmin - 1e-9, ymin - 1e-9]) or np.any(high > [xmax + 1e-9, ymax + 1e-9]):
            raise ValueError(f"{spec['name']} extends outside the pickup area")
        for other in range(index):
            if overlaps(positions[index], positions[other], half[index], half[other]):
                raise ValueError(
                    f"Leave at least 8 mm between {spec['name']} and {specs[other]['name']}"
                )
    return {
        "name": definition["name"].strip(),
        "seed": definition.get("seed", 0),
        "target": definition["target"],
        "objects": [
            {"id": spec["id"], "x": float(xy[0]), "y": float(xy[1])}
            for spec, xy in zip(specs, positions)
        ],
    }


def shuffle_layout(cache, definition, seed):
    specs = _specs(cache, {**definition, "seed": seed})
    half = np.asarray([spec["dimensions"][:2] for spec in specs]) / 2
    xmin, xmax, ymin, ymax = WORKSPACE
    lower, upper = np.array([xmin, ymin]) + half, np.array([xmax, ymax]) - half
    if np.any(lower > upper):
        raise ValueError("An object is too large for the pickup area; resize it in the importer")
    rng = np.random.default_rng(seed)
    # Largest footprints first; restart if a partial arrangement blocks the remainder.
    order = np.argsort(-np.prod(half, axis=1), kind="stable")
    for _ in range(40):
        positions = {}
        for index in order:
            for _ in range(250):
                point = rng.uniform(lower[index], upper[index])
                if not any(
                    overlaps(point, xy, half[index], half[j]) for j, xy in positions.items()
                ):
                    positions[index] = point
                    break
            else:
                break
        if len(positions) == len(specs):
            return validate_layout(
                cache,
                {
                    **definition,
                    "seed": seed,
                    "objects": [
                        {"id": spec["id"], "x": float(positions[i][0]), "y": float(positions[i][1])}
                        for i, spec in enumerate(specs)
                    ],
                },
            )
    raise ValueError("These objects do not fit with pickup clearance. Remove or resize an object")


def layout_path(cache, identifier):
    return Path(cache) / "layouts" / f"{UUID(str(identifier))}.json"


def saved_layouts(cache):
    folder = Path(cache) / "layouts"
    result = []
    for path in folder.glob("*.json"):
        try:
            if path.stat().st_size > 100_000:
                continue
            document = json.loads(path.read_text())
            if (
                document["schema"] == "panda-layout-v1"
                and layout_path(cache, document["id"]) == path
            ):
                result.append(document)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return sorted(result, key=lambda item: item.get("updated", ""), reverse=True)


def save_layout(cache, definition, identifier=None):
    layout = validate_layout(cache, definition)
    identifier = str(UUID(str(identifier))) if identifier else str(uuid4())
    path = layout_path(cache, identifier)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        **layout,
        "id": identifier,
        "schema": "panda-layout-v1",
        "updated": datetime.now(timezone.utc).isoformat(),
    }
    handle, temporary = tempfile.mkstemp(prefix=".layout-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(handle, "w") as out:
            json.dump(document, out, indent=2)
            out.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return document
