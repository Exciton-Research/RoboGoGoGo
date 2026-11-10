"""Single/multi-arm embodiments, isolated commands, and aligned VLA data."""

import io
import json
import zipfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fastapi.testclient import TestClient
from fixtures import models
from fixtures.api import create_app
from test_vla import png_observation

from robogogogo import control, urdf
from robogogogo import spatial as sp
from robogogogo.embodiment import resolve_arms


@pytest.mark.parametrize("count", [1, 2, 3])
def test_arm_groups_cover_multiple_branches(count):
    model = urdf.loads(models.multi_arm_urdf(count)).model
    arms = resolve_arms(model)
    assert len(arms) == count
    assert sorted(i for arm in arms for i in arm.joint_indices) == list(range(count * 6))
    pose = sp.forward_kinematics(model, model.home)
    assert len({tuple(np.asarray(pose.position[arm.body])) for arm in arms}) == count


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    with TestClient(
        create_app(project_dir=tmp_path_factory.mktemp("multi-arm"), initial_scene="dual_arm")
    ) as client:
        yield client


def test_selected_arm_joint_command_preserves_other_goals(client):
    before = client.get("/api/snapshot").json()
    groups = before["model"]["arms"]
    assert [g["name"] for g in groups] == ["left_arm", "right_arm"]
    assert before["model"]["dof"] == 12
    left, right = [g["joint_indices"] for g in groups]
    values = np.asarray(before["state"]["goal"])[left] + 0.05
    response = client.post(
        "/api/command", json=dict(op="joints", arm="left_arm", values=values.tolist())
    )
    response.raise_for_status()
    after = response.json()
    np.testing.assert_allclose(
        np.asarray(after["state"]["goal"])[right], np.asarray(before["state"]["goal"])[right]
    )
    assert after["state"]["q"] == before["state"]["q"]
    assert (
        client.post(
            "/api/command", json=dict(op="target", arm="missing", values=[1, 0, 1])
        ).status_code
        == 422
    )


def test_multigoal_ik_supports_shared_ancestor():
    builder = sp.ModelBuilder()
    root = builder.add_link("torso", axis=(0, 0, 1), inertia=np.eye(3) * 0.1)
    for name, offset in [("left", -0.4), ("right", 0.4)]:
        builder.add_link(
            name,
            parent=root,
            joint="prismatic",
            axis=(1, 0, 0),
            origin=(0, offset, 0),
            inertia=np.eye(3) * 0.1,
            limits=(-1, 1),
        )
    model = builder.build()
    arms = resolve_arms(model)
    assert arms[0].joint_indices[0] == arms[1].joint_indices[0] == 0
    desired = jnp.array([0.15, 0.2, 0.1])
    targets = jnp.stack([sp.point_position(model, desired, arm.body) for arm in arms])
    q, errors = jax.jit(
        lambda goals: control.inverse_kinematics_targets(
            model, goals, tuple(arm.body for arm in arms), model.home
        )
    )(targets)
    assert np.max(errors) < 1e-5
    assert np.isfinite(q).all()


def test_dual_arm_episode_export_and_restore(client, tmp_path):
    snapshot = client.post("/api/vla/reset", json={}).json()
    arms = snapshot["episode"]["arms"]
    assert len(snapshot["episode"]["targets"]) == 2
    assert snapshot["episode"]["task"] == "multi_arm_reach"

    def observe(snapshot):
        result = client.post(
            "/api/vla/observe", json=dict(image=png_observation(), sequence=snapshot["sequence"])
        )
        result.raise_for_status()
        return result.json()

    before_q = np.asarray(snapshot["state"]["q"])
    observed = observe(snapshot)
    action = [0.03] + [0] * 5 + [-0.03] + [0] * 5
    response = client.post(
        "/api/vla/act", json=dict(delta=action, observation=observed["observation"])
    )
    response.raise_for_status()
    snapshot = response.json()
    assert not snapshot["episode"]["success"]
    assert len(snapshot["state"]["target_errors"]) == 2
    assert np.linalg.norm(np.asarray(snapshot["state"]["q"])[:6] - before_q[:6]) > 0
    assert np.linalg.norm(np.asarray(snapshot["state"]["q"])[6:] - before_q[6:]) > 0
    observe(snapshot)
    saved = client.post("/api/vla/save", json={}).json()
    archive_path = tmp_path / "dual.zip"
    archive_path.write_bytes(client.get(f"/api/vla/episodes/{saved['id']}").content)
    from robogogogo.datasets import iter_transitions

    transition = next(iter_transitions(archive_path))
    assert len(transition.action) == 12
    assert len(transition.observation.arms) == 2
    assert set(transition.observation.tool_positions) == {"left_arm", "right_arm"}
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("episode.json"))
        assert manifest["action_spec"]["arms"] == arms
        assert len(manifest["transitions"][0]["target_errors"]) == 2
    checkpoint = client.post("/api/projects", json={"name": "Dual arms"}).json()["project"]
    restored = client.post(f"/api/projects/{checkpoint['id']}/load", json={}).json()
    assert restored["state"]["targets"] == snapshot["state"]["targets"]
    assert restored["model"]["arms"] == arms
    with np.load(io.BytesIO(client.get("/api/export").content), allow_pickle=False) as data:
        assert data["arm_targets"].shape == (1, 2, 3)


def test_reference_reaches_every_arm(client):
    snapshot = client.post("/api/vla/reset", json={}).json()
    for _ in range(50):
        observed = client.post(
            "/api/vla/observe", json=dict(image=png_observation(), sequence=snapshot["sequence"])
        ).json()
        if observed["snapshot"]["episode"]["done"]:
            snapshot = observed["snapshot"]
            break
        reference = client.get("/api/vla/reference-action")
        reference.raise_for_status()
        response = client.post(
            "/api/vla/act",
            json=dict(
                delta=reference.json()["delta"],
                observation=observed["observation"],
                source="scripted_reference",
            ),
        )
        response.raise_for_status()
        snapshot = response.json()
    assert snapshot["episode"]["success"]
    assert all(value < 0.02 for value in snapshot["state"]["target_errors"].values())


def test_one_reached_arm_is_not_task_success():
    from robogogogo.vla import Episode

    episode = Episode(
        "Reach both",
        [0, 0, 0],
        joint_names=["left", "right"],
        joint_kinds=["revolute"] * 2,
        dt=0.002,
        substeps=5,
        arms=[dict(name="left", body=0), dict(name="right", body=1)],
        targets={"left": [0, 0, 0], "right": [0, 0, 0]},
    )
    before = {"state": dict(time=0)}
    after = {"state": dict(positions=[[0, 0, 0], [0.1, 0, 0]], q=[0, 0], qd=[0, 0], time=0.1)}
    episode.append_action(
        requested=[0, 0],
        applied=[0, 0],
        ticks=10,
        source="manual",
        previous=before,
        following=after,
    )
    assert not episode.success
    assert episode.actions[-1]["target_errors"] == {"left": 0, "right": 0.1}


def test_switching_embodiments_starts_usable_episode(client):
    for scene, count in [("arm", 1), ("dual_arm", 2)]:
        response = client.post("/api/command", json=dict(op="scene", scene=scene))
        response.raise_for_status()
        snapshot = response.json()
        assert len(snapshot["model"]["arms"]) == count
        assert not snapshot["episode"]["done"]
        capture = client.post(
            "/api/vla/observe",
            json=dict(
                images={c["name"]: png_observation() for c in snapshot["episode"]["cameras"]},
                sequence=snapshot["sequence"],
            ),
        )
        capture.raise_for_status()
