"""Layout placement, persistence, reset, and episode compatibility."""

import os

import numpy as np
import pytest

from robogogogo.demo import layouts


@pytest.fixture
def layout_assets(monkeypatch):
    specs = {
        "box": {"id": "box", "name": "Box", "dimensions": [0.04, 0.09, 0.12]},
        "bottle": {"id": "bottle", "name": "Bottle", "dimensions": [0.05, 0.07, 0.14]},
    }
    monkeypatch.setattr(layouts, "require_object", lambda cache, identifier: specs[identifier])
    return {
        "name": "Kitchen table",
        "seed": 5,
        "target": "bottle",
        "objects": [{"id": "box", "x": 0.35, "y": -0.22}, {"id": "bottle", "x": 0.60, "y": -0.18}],
    }


def test_saved_layout_roundtrip_and_update_preserve_selected_target(tmp_path, layout_assets):
    saved = layouts.save_layout(tmp_path, layout_assets)
    assert layouts.saved_layouts(tmp_path) == [saved]
    assert saved["target"] == "bottle"
    assert saved["objects"] == layout_assets["objects"]
    changed = {**layout_assets, "name": "Evening table"}
    replacement = layouts.save_layout(tmp_path, changed, saved["id"])
    assert replacement["id"] == saved["id"]
    assert layouts.saved_layouts(tmp_path) == [replacement]


def test_shuffle_is_repeatable_and_has_clearance(tmp_path, layout_assets):
    first = layouts.shuffle_layout(tmp_path, layout_assets, 13)
    assert first == layouts.shuffle_layout(tmp_path, layout_assets, 13)
    assert first["objects"] != layouts.shuffle_layout(tmp_path, layout_assets, 14)["objects"]
    assert first["target"] == "bottle"
    assert layouts.validate_layout(tmp_path, first) == first


@pytest.mark.parametrize("problem", ["overlap", "boundary", "duplicate", "target", "nan"])
def test_invalid_layout_does_not_overwrite_a_saved_table(tmp_path, layout_assets, problem):
    saved = layouts.save_layout(tmp_path, layout_assets)
    items = [dict(item) for item in layout_assets["objects"]]
    changed = {**layout_assets, "objects": items}
    if problem == "overlap":
        items[1].update(x=items[0]["x"], y=items[0]["y"])
    elif problem == "boundary":
        items[0]["x"] = 0.21
    elif problem == "duplicate":
        items[1]["id"] = "box"
    elif problem == "target":
        changed["target"] = "absent"
    else:
        items[0]["y"] = float("nan")
    with pytest.raises(ValueError):
        layouts.save_layout(tmp_path, changed, saved["id"])
    assert layouts.saved_layouts(tmp_path) == [saved]


def test_custom_layout_reset_and_recording_replay(tmp_path):
    assets = os.environ.get("ROBOGOGOGO_VISION_ASSETS")
    if not assets:
        pytest.skip("Set ROBOGOGOGO_VISION_ASSETS for native layout and recording checks")
    pytest.importorskip("mujoco")
    from robogogogo.demo.environment import PandaEnv
    from robogogogo.demo.recording import Episode, Recording

    definition = {
        "name": "Fruit and toy",
        "seed": 2,
        "target": "lemon",
        "objects": [
            {"id": "rubiks-cube", "x": 0.34, "y": -0.22},
            {"id": "lemon", "x": 0.60, "y": -0.17},
        ],
    }
    with PandaEnv(assets, scene_layout=definition) as env:
        assert env.collection_style == "bin"
        assert env.object["id"] == "lemon"
        original = env.data.xpos[env.bodies].copy()
        env.reset(seed=2)
        np.testing.assert_array_equal(original, env.data.xpos[env.bodies])
        np.testing.assert_allclose(original[:, :2], [[0.34, -0.22], [0.60, -0.17]], atol=0.003)
        assert env.snapshot()["table_layout"] == definition
        recording = Recording(tmp_path, env, seed=2, images=False)
        action = env.previous_action.copy()
        recording.append(env, action, env.step(action), "manual")
        recording.save(env, "manual")
        final = env.capture_state()["integration"].copy()
        manifest = recording.manifest
    with PandaEnv(assets, scene_layout=manifest["scene"]["table_layout"]) as replay:
        episode = Episode(tmp_path / f"{recording.id}.zip", replay)
        replay.restore_state(episode.state(episode.count))
        np.testing.assert_array_equal(replay.capture_state()["integration"], final)


def test_layout_api_rejects_changes_atomically_and_protects_recordings(tmp_path):
    assets = os.environ.get("ROBOGOGOGO_VISION_ASSETS")
    if not assets:
        pytest.skip("Set ROBOGOGOGO_VISION_ASSETS for native scene API checks")
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from robogogogo.demo.__main__ import create_app

    definition = {
        "name": "API table",
        "seed": 4,
        "target": "lemon",
        "objects": [
            {"id": "lemon", "x": 0.35, "y": -0.22},
            {"id": "orange", "x": 0.60, "y": -0.18},
        ],
    }
    with TestClient(
        create_app(assets, static_dir=tmp_path, recordings=tmp_path / "episodes"),
        base_url="http://localhost",
    ) as client:
        response = client.post("/api/demo/layouts/apply", json=definition)
        assert response.status_code == 200, response.text
        current = response.json()
        assert current["table_layout"] == definition
        bad = {
            **definition,
            "objects": [
                {"id": "lemon", "x": 0.35, "y": -0.22},
                {"id": "orange", "x": 0.35, "y": -0.22},
            ],
        }
        assert client.post("/api/demo/layouts/apply", json=bad).status_code == 422
        assert client.get("/api/demo/state").json()["episode_id"] == current["episode_id"]
        assert (
            client.post("/api/demo/record", json={"command": "start", "images": False}).status_code
            == 200
        )
        blocked = client.post("/api/demo/layouts/apply", json=definition)
        assert blocked.status_code == 422 and "recording" in blocked.json()["detail"]
        assert client.get("/api/demo/state").json()["episode_id"] == current["episode_id"]
        assert client.post("/api/demo/record", json={"command": "discard"}).status_code == 200
