"""Reference geometry, body/arm separation and camera frame consistency."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fastapi.testclient import TestClient

from robogogogo import spatial as sp
from robogogogo.control import inverse_kinematics_targets
from robogogogo.embodiment import resolve_arms
from robogogogo.robot.cameras import mounted_rig, world_camera_pose
from robogogogo.robot.mobile_design import mobile_arm_definitions, mobile_dual_arm_urdf, mobile_home
from robogogogo.studio.api import create_app
from robogogogo.urdf import loads


@pytest.fixture(scope="module")
def model():
    m = loads(mobile_dual_arm_urdf()).model
    return replace(m, home=jnp.asarray(mobile_home(m)))


def test_structure_has_four_steerable_wheels_foldable_body_and_seven_joint_arms(model):
    arms = resolve_arms(model, mobile_arm_definitions())
    assert len(model.home) == 31
    assert len([name for name in model.names if name.endswith("_wheel")]) == 4
    assert len([name for name in model.names if name.endswith("_steer")]) == 4
    assert all(len(a.joint_indices) == 7 for a in arms)
    assert not set(arms[0].joint_indices) & set(arms[1].joint_indices)
    torso_indices = {
        model.indices[model.names.index(n)]
        for n in (
            "base_x",
            "base_y",
            "base_link",
            "torso_yaw",
            "torso_lower",
            "torso_upper",
            "torso",
        )
    }
    assert all(not torso_indices.intersection(a.joint_indices) for a in arms)
    with pytest.raises(ValueError, match="ancestor"):
        resolve_arms(model, [dict(name="left", tool="left_tool", root="right_tool")])


def test_torso_postures_lower_and_extend_upper_body_without_tilting_head(model):
    poses = {
        name: sp.forward_kinematics(model, jnp.asarray(mobile_home(model, name)))
        for name in ("standing", "low", "reach")
    }
    chest, head = model.names.index("torso"), model.names.index("head")
    assert poses["low"].position[chest, 2] < poses["standing"].position[chest, 2] - 0.35
    assert poses["reach"].position[chest, 0] > poses["standing"].position[chest, 0] + 0.20
    for posture, pose in poses.items():
        np.testing.assert_allclose(pose.rotation[head], np.eye(3), atol=1e-6)
        q = mobile_home(model, posture)
        assert np.all(q >= model.lower) and np.all(q <= model.upper)


def test_base_and_torso_transform_all_four_cameras_consistently(model):
    arms = resolve_arms(model, mobile_arm_definitions())
    cameras = mounted_rig(model, arms)
    q = model.home
    original = sp.forward_kinematics(model, q)
    shifted = q.at[model.indices[model.names.index("base_x")]].add(0.4)
    moved = sp.forward_kinematics(model, shifted)
    low = sp.forward_kinematics(model, jnp.asarray(mobile_home(model, "low")))
    for camera in cameras:
        camera["body"] = model.names.index(camera["mount"])

        def optical(pose):
            return world_camera_pose(camera, dict(positions=pose.position, rotations=pose.rotation))

        a, b, c = optical(original), optical(moved), optical(low)
        np.testing.assert_allclose(np.array(b["position"]) - a["position"], [0.4, 0, 0], atol=1e-6)
        assert c["position"][2] < a["position"][2] - 0.35
        np.testing.assert_allclose(a["rotation"], b["rotation"], atol=1e-6)


def test_arm_only_ik_does_not_drive_base_torso_or_wheels(model):
    arms = resolve_arms(model, mobile_arm_definitions())
    indices = sorted({j for arm in arms for j in arm.joint_indices})
    mask = jnp.zeros_like(model.home).at[jnp.array(indices)].set(1)
    pose = sp.forward_kinematics(model, model.home)
    bodies = tuple(a.body for a in arms)
    targets = jnp.asarray([pose.position[b] for b in bodies]) + jnp.array([0.02, 0, 0])
    solve = jax.jit(
        lambda q: inverse_kinematics_targets(model, targets, bodies, q, joint_mask=mask)
    )
    q, error = solve(model.home)
    frozen = np.flatnonzero(np.asarray(mask) == 0)
    np.testing.assert_array_equal(q[frozen], model.home[frozen])
    assert np.max(error) < 0.005


def test_default_studio_mobile_postures_dynamics_checkpoint_and_task_boundary(tmp_path):
    with TestClient(create_app(project_dir=tmp_path)) as client:
        snapshot = client.get("/api/snapshot").json()
        assert snapshot["model"]["scene"] == "mobile_dual_arm"
        assert snapshot["model"]["dof"] == 31
        assert [len(a["joint_indices"]) for a in snapshot["model"]["arms"]] == [7, 7]
        assert len(snapshot["episode"]["cameras"]) == 4
        result = client.post("/api/command", json={"op": "posture", "posture": "low"})
        result.raise_for_status()
        goal = result.json()["state"]["goal"]
        assert goal != snapshot["state"]["q"]
        after = client.post("/api/command", json={"op": "step"}).json()
        assert np.isfinite(after["state"]["q"]).all()
        assert after["state"]["q"] != snapshot["state"]["q"]
        saved = client.post("/api/projects", json={"name": "Mobile concept"}).json()
        project = saved["project"]
        assert all(a["root"] for a in project["arms"])
        restored = client.post(f"/api/projects/{project['id']}/load").json()
        np.testing.assert_allclose(restored["state"]["goal"], after["state"]["goal"])
        blocked = client.post("/api/tasks/plan", json={"kind": "handover"})
        assert blocked.status_code == 422
        assert "stability" in blocked.json()["detail"]
