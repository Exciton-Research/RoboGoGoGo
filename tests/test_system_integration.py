"""Capture, supervised action, perception and opt-in depth export together."""

import base64
import io
import zipfile

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from robogogogo.datasets import iter_transitions
from robogogogo.server import create_app

cv = pytest.importorskip("cv2")


def test_stereo_and_rule_control_roundtrip(tmp_path):
    with TestClient(create_app(project_dir=tmp_path, initial_scene="arm")) as client:
        snapshot = client.post("/api/episodes/reset", json={"camera_preset": "head"}).json()
        image = np.full((240, 320, 3), 128, np.uint8)
        _, png = cv.imencode(".png", image)
        url = "data:image/png;base64," + base64.b64encode(png).decode()
        encoded = base64.b64encode(np.ones((240, 320), dtype="<f4").tobytes()).decode()

        def capture(snapshot):
            response = client.post(
                "/api/episodes/observe",
                json={
                    "sequence": snapshot["sequence"],
                    "images": {c["name"]: url for c in snapshot["episode"]["cameras"]},
                    "ground_truth_depth": {
                        c["name"]: encoded for c in snapshot["episode"]["cameras"]
                    },
                },
            )
            assert response.status_code == 200, response.text
            return response.json()

        first = capture(snapshot)
        report = client.post("/api/perception/stereo", json={"observation": 0})
        assert report.status_code == 200, report.text
        assert report.json()["metrics"]["mae_m"] is None
        assert report.json()["baseline_m"] == pytest.approx(0.064)
        action = client.get("/api/control/rule-based-action")
        assert action.status_code == 200, action.text
        assert action.json()["source"] == "rule_based"
        after = client.post(
            "/api/episodes/act",
            json={
                "observation": first["observation"],
                "delta": action.json()["delta"],
                "source": "rule_based",
            },
        ).json()
        capture(after)
        saved = client.post("/api/episodes/save", json={}).json()
        path = tmp_path / "episodes" / f"{saved['id']}.zip"
        ordinary = list(iter_transitions(path))
        privileged = list(iter_transitions(path, include_ground_truth=True))
        assert ordinary[0].source == "rule_based"
        assert not ordinary[0].observation.ground_truth_depth
        np.testing.assert_equal(
            privileged[0].observation.ground_truth_depth["head_left"], np.ones((240, 320))
        )
        with zipfile.ZipFile(path) as archive:
            depth = np.load(
                io.BytesIO(archive.read("depth/head_left/000000.npy")), allow_pickle=False
            )
            assert depth.dtype == np.float32
