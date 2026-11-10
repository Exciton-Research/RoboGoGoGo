"""The installed product admits one embodiment, independent of numerical fixtures."""

import json
import zipfile

import pytest

from robogogogo.learning.datasets import inspect_dataset
from robogogogo.studio.api import create_app


def test_unsupported_product_routes_are_absent(tmp_path):
    paths = {route.path for route in create_app(project_dir=tmp_path).routes}
    assert "/api/robot" not in paths
    assert not any(path.startswith(("/api/tasks/", "/api/structure")) for path in paths)


@pytest.mark.parametrize("robot", [None, {"id": "other", "revision": "concept-1"}])
def test_learning_rejects_other_or_unversioned_robot(tmp_path, robot):
    path = tmp_path / "episode.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("episode.json", json.dumps({"robot": robot}))
    with pytest.raises(ValueError, match="RG Mobile Duo"):
        inspect_dataset(path)
