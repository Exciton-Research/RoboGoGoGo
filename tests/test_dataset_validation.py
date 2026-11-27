"""Corrupt observations and incompatible action contracts must not become learning inputs."""

import base64
import io
import json
import struct
import zipfile

import pytest
from PIL import Image

from robogogogo.data.datasets import iter_transitions
from robogogogo.data.images import decode_camera_png
from robogogogo.learning.datasets import inspect_dataset
from robogogogo.robot.definition import ROBOT_ID, ROBOT_REVISION, SIMULATION_JOINT_NAMES

CAMERA = dict(name="head_left", width=32, height=32)


def png(mode="RGB"):
    output = io.BytesIO()
    Image.new(mode, (32, 32)).save(output, format="PNG")
    return output.getvalue()


def archive(tmp_path, *, image=None, action_changes=None, cameras=None):
    camera_list = [CAMERA] if cameras is None else cameras
    manifest = dict(
        id="00000000-0000-0000-0000-000000000001",
        format="robogogogo-episode-v2",
        environment="simulation",
        robot=dict(id=ROBOT_ID, revision=ROBOT_REVISION),
        cameras=camera_list,
        action_spec=dict(
            representation="joint_position_delta",
            dimensions=31,
            names=list(SIMULATION_JOINT_NAMES),
            units=["m", "m"] + ["rad"] * 29,
            frame="joint_coordinates",
            normalization="none",
        ),
        observations=[
            dict(
                instruction="Reach the target.",
                time=i * 0.1,
                joint_position=[0.0] * 31,
                joint_velocity=[0.0] * 31,
                images={"head_left": f"frames/head_left/{i:06d}.png"},
            )
            for i in range(2)
        ],
        transitions=[
            dict(
                observation=0,
                next_observation=1,
                time=0.0,
                next_time=0.1,
                applied_delta=[0.0] * 31,
                reward=0.0,
                terminated=False,
                truncated=False,
                source="manual",
            )
        ],
    )
    manifest["action_spec"].update(action_changes or {})
    path = tmp_path / "episode.zip"
    with zipfile.ZipFile(path, "w") as output:
        output.writestr("episode.json", json.dumps(manifest))
        for i in range(2):
            output.writestr(f"frames/head_left/{i:06d}.png", png() if image is None else image)
    return path


@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
def test_valid_camera_images_and_dataset_round_trip(tmp_path, mode):
    data = png(mode)
    url = "data:image/png;base64," + base64.b64encode(data).decode()
    assert decode_camera_png(url, CAMERA) == data
    report = inspect_dataset(archive(tmp_path, image=data))
    assert report["transitions"] == 1
    assert report["cameras"] == ["head_left"]


@pytest.mark.parametrize("kind", ["header_only", "truncated", "bad_checksum", "invalid_pixels"])
def test_capture_rejects_corrupt_png_data(kind):
    data = png()
    if kind == "header_only":
        data = b"\x89PNG\r\n\x1a\n" + b"\0" * 8 + struct.pack(">II", 32, 32)
    elif kind == "truncated":
        data = data[:-12]
    elif kind == "bad_checksum":
        damaged = bytearray(data)
        damaged[-13] ^= 1
        data = bytes(damaged)
    else:
        import zlib

        # Valid chunk checksums cannot make an invalid compressed pixel stream decodable.
        start = data.index(b"IDAT")
        length = struct.unpack(">I", data[start - 4 : start])[0]
        payload = b"IDAT" + b"x" * length
        data = (
            data[:start]
            + payload
            + struct.pack(">I", zlib.crc32(payload))
            + data[start + 4 + length + 4 :]
        )
    url = "data:image/png;base64," + base64.b64encode(data).decode()
    with pytest.raises(ValueError, match="PNG"):
        decode_camera_png(url, CAMERA)


def test_dataset_reader_rejects_corrupted_saved_frames(tmp_path):
    path = archive(tmp_path, image=b"not an image")
    with pytest.raises(ValueError, match="PNG"):
        list(iter_transitions(path))
    with pytest.raises(ValueError, match="PNG"):
        inspect_dataset(path)


@pytest.mark.parametrize("cameras", [[], [CAMERA, CAMERA]])
def test_dataset_reader_rejects_empty_or_duplicate_camera_streams(tmp_path, cameras):
    with pytest.raises(ValueError, match="camera"):
        list(iter_transitions(archive(tmp_path, cameras=cameras)))


@pytest.mark.parametrize(
    "changes",
    [
        {"units": ["rad"] * 31},
        {"units": None},
        {"frame": "world"},
        {"normalization": "standardized"},
    ],
)
def test_learning_rejects_incompatible_action_semantics(tmp_path, changes):
    with pytest.raises(ValueError, match="units|frame|normalization"):
        inspect_dataset(archive(tmp_path, action_changes=changes))


@pytest.mark.parametrize("manifest", [[], None, {"action_spec": None}])
def test_malformed_manifest_is_an_actionable_validation_error(tmp_path, manifest):
    from robogogogo.runtime.workflow import RobotWorkflow

    workflow = RobotWorkflow(tmp_path)
    workflow.episodes.directory.mkdir()
    path = workflow.episodes.archive("00000000-0000-0000-0000-000000000001")
    with zipfile.ZipFile(path, "w") as output:
        output.writestr("episode.json", json.dumps(manifest))
    with pytest.raises(ValueError):
        workflow.validate_episode(path.stem)
    with pytest.raises(ValueError, match="manifest|specification"):
        list(iter_transitions(path))
