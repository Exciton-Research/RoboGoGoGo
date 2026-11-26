"""The installed product admits one embodiment, independent of numerical fixtures."""

import json
import zipfile

import pytest

from robogogogo.learning.datasets import inspect_dataset
from robogogogo.studio.api import create_app


def test_unsupported_product_routes_are_absent(tmp_path):
    paths = set(create_app(project_dir=tmp_path).openapi()["paths"])
    assert "/api/robot" not in paths
    assert not any(path.startswith(("/api/tasks/", "/api/structure")) for path in paths)


@pytest.mark.parametrize("robot", [None, {"id": "other", "revision": "concept-1"}])
def test_learning_rejects_other_or_unversioned_robot(tmp_path, robot):
    path = tmp_path / "episode.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("episode.json", json.dumps({"robot": robot}))
    with pytest.raises(ValueError, match="RG Mobile Duo"):
        inspect_dataset(path)


def test_canonical_joint_order_matches_model():
    from robogogogo.robot.definition import SIMULATION_JOINT_NAMES
    from robogogogo.robot.mobile_design import mobile_dual_arm_urdf
    from robogogogo.urdf import loads

    model = loads(mobile_dual_arm_urdf()).model
    assert (
        tuple(name for name, i in zip(model.names, model.indices) if i >= 0)
        == SIMULATION_JOINT_NAMES
    )


def test_learning_rejects_reordered_robot_joints(tmp_path):
    from robogogogo.robot.definition import ROBOT_ID, ROBOT_REVISION, SIMULATION_JOINT_NAMES

    path = tmp_path / "episode.zip"
    manifest = dict(
        robot=dict(id=ROBOT_ID, revision=ROBOT_REVISION),
        action_spec=dict(dimensions=31, names=list(reversed(SIMULATION_JOINT_NAMES))),
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("episode.json", json.dumps(manifest))
    with pytest.raises(ValueError, match="joint ordering"):
        inspect_dataset(path)


def test_mobile_trajectory_preserves_robot_identity(tmp_path):
    import numpy as np

    from robogogogo.robot.definition import ROBOT_ID, ROBOT_REVISION
    from robogogogo.studio.workspace import Studio

    studio = Studio(tmp_path)
    path = studio.session.export(tmp_path / "trajectory.npz")
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
    assert metadata["scene"] == "mobile_dual_arm"
    assert metadata["robot"] == dict(id=ROBOT_ID, revision=ROBOT_REVISION)
