import json
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from robogogogo.hardware import CameraInfoSample, CameraSample, JointSample
from robogogogo.hardware.calibration import CalibrationProfile
from robogogogo.hardware.perception import HardwarePerception
from robogogogo.hardware.synchronization import Synchronizer

XML = """<robot name="measured_rig"><link name="base"/><link name="wrist"/>
<joint name="slide" type="prismatic"><parent link="base"/><child link="wrist"/>
<axis xyz="1 0 0"/></joint></robot>"""


def document():
    def camera(frame, mount, x):
        transform = np.eye(4)
        transform[0, 3] = x
        return dict(
            frame_id=frame,
            width=192,
            height=96,
            intrinsics=[[120.0, 0, 96], [0, 120.0, 48], [0, 0, 1]],
            distortion=[0.0, 0, 0, 0, 0],
            distortion_model="plumb_bob",
            mount_link=mount,
            mount_from_optical=transform.tolist(),
            clock_domain="robot",
            time_offset_s=0.0,
            time_uncertainty_s=0.0001,
            reprojection_error_px=0.2,
            extrinsic_uncertainty_m=0.0002,
            rotation_uncertainty_rad=0.00005,
        )

    return dict(
        schema_version=1,
        revision="lab-measurement-001",
        calibrated_at="2026-11-09T12:00:00Z",
        fixed_base=True,
        world_frame="lab",
        clock_domain="robot",
        robot_urdf="robot.urdf",
        world_from_base=np.eye(4).tolist(),
        joint_speed_limits={"slide": 0.1},
        max_scene_speed_m_s=0.1,
        max_skew_s=0.01,
        max_joint_gap_s=0.04,
        cameras={
            "head": camera("head_optical", "base", 0),
            "wrist": camera("wrist_optical", "wrist", 0.064),
        },
        stereo_pairs=[["head", "wrist"]],
    )


def profile():
    return CalibrationProfile(document(), XML)


def joint(t, q=0):
    return JointSample(t, ("unrelated", "slide"), (2.0, q), clock_domain="robot")


def frame(name, t, data=b""):
    return CameraSample(name, t, data, "image/png", name + "_optical", "robot")


def ready(sync, t=1.0, q=0):
    sync.add(joint(t, q), 0.0)
    sync.add(joint(t + 0.02, q), 0.01)
    sync.add(frame("head", t + 0.01), 0.02)
    sync.add(frame("wrist", t + 0.012), 0.03)


def test_joint_interpolation_reorders_names_and_camera_poses_follow_exposure():
    p = profile()
    sync = Synchronizer(p)
    sync.add(joint(1.0, 0), 0)
    sync.add(joint(1.02, 0.001), 0.01)
    sync.add(frame("head", 1.01), 0.02)
    sync.add(frame("wrist", 1.012), 0.02)
    bundle = sync.next(0.03)
    assert bundle is not None
    assert bundle.positions["head"]["slide"] == pytest.approx(0.0005)
    assert bundle.positions["wrist"]["slide"] == pytest.approx(0.0006)
    assert p.pose("head", bundle.positions["head"])["position"][0] == 0
    assert p.pose("wrist", bundle.positions["wrist"])["position"][0] == pytest.approx(0.0646)
    assert bundle.skew_s == pytest.approx(0.002)
    assert sync.next(0.04) is None  # Frames consumed once.


def test_extrapolation_and_large_joint_gaps_are_rejected():
    sync = Synchronizer(profile())
    sync.add(joint(1), 0)
    with pytest.raises(ValueError, match="extrapolation"):
        sync.interpolate(1.01)
    sync.add(joint(1.1), 0.1)
    with pytest.raises(ValueError, match="gap"):
        sync.interpolate(1.01)


def test_clock_domains_frames_staleness_and_skew_are_enforced():
    sync = Synchronizer(profile())
    sync.add(replace(joint(1), clock_domain="host"), 0)
    assert not sync.joints
    sync.add(replace(frame("head", 1), frame_id="wrong"), 0)
    assert not sync.frames["head"]
    ready(sync)
    assert sync.next(1) is None
    assert sync.dropped == 4 and sync.rejected == 2
    sync = Synchronizer(profile())
    ready(sync)
    sync.frames["wrist"].clear()
    sync.add(frame("wrist", 1.1), 0.03)
    assert sync.next(0.04) is None and "skew" in sync.reason


def test_small_arrival_reordering_and_continuous_joint_wrap():
    d = document()
    d["joint_speed_limits"]["slide"] = 1.0
    p = CalibrationProfile(d, XML.replace("prismatic", "continuous"))
    sync = Synchronizer(p)
    sync.add(joint(1.02, -np.pi + 0.005), 0)
    sync.add(joint(1.0, np.pi - 0.005), 0.01)
    q, _ = sync.interpolate(1.01)
    assert abs(q["slide"] - np.pi) < 1e-8
    sync.add(joint(0.0), 0.02)
    assert sync.fault and sync.next(0.03) is None


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(fixed_base=False),
        lambda d: d["cameras"]["head"].update(mount_link="missing"),
        lambda d: d["cameras"]["head"].update(clock_domain="other_device"),
        lambda d: d["cameras"]["head"].update(reprojection_error_px=5),
        lambda d: d["cameras"]["head"].update(time_uncertainty_s=0.1),
        lambda d: d["cameras"]["head"].update(intrinsics=np.zeros((3, 3)).tolist()),
        lambda d: d.update(joint_speed_limits={}),
    ],
)
def test_bad_calibration_never_activates(change):
    d = document()
    change(d)
    with pytest.raises(ValueError):
        CalibrationProfile(d, XML)


def test_profile_persistence_digest_and_hardware_urdf_mesh_metadata(tmp_path):
    d = document()
    (tmp_path / "profile.json").write_text(json.dumps(d))
    (tmp_path / "robot.urdf").write_text(XML)
    loaded = CalibrationProfile.load(tmp_path / "profile.json")
    assert loaded.digest == profile().digest
    altered = deepcopy(d)
    altered["revision"] = "new-measurement"
    assert CalibrationProfile(altered, XML).digest != loaded.digest
    mesh_xml = XML.replace(
        '<link name="base"/>',
        '<link name="base"><visual><geometry><mesh filename="robot.stl"/></geometry></visual></link>',
    )
    assert CalibrationProfile(d, mesh_xml).tree.names == ("slide",)


def textured_frames():
    cv = pytest.importorskip("cv2")
    left = np.random.default_rng(7).integers(0, 256, (96, 192, 3), dtype=np.uint8)
    right = np.zeros_like(left)
    right[:, :-32] = left[:, 32:]
    return tuple(cv.imencode(".png", image)[1].tobytes() for image in (left, right))


def test_calibrated_stereo_reaches_memory_and_audits_actual_capture_joints(tmp_path):
    pipeline = HardwarePerception(profile(), clock=lambda: 0.1, audit_dir=tmp_path)
    left, right = textured_frames()
    pipeline.ingest(joint(1))
    pipeline.ingest(joint(1.02))
    pipeline.ingest(frame("head", 1.01, left))
    pipeline.ingest(frame("wrist", 1.012, right))
    pipeline.step()
    status = pipeline.snapshot()
    assert status["state"] == "observing"
    assert status["scene_memory"]["surfaces"] > 0
    assert status["calibration_digest"] == profile().digest
    records = list(tmp_path.glob("*/observations.jsonl"))
    record = json.loads(records[0].read_text())
    assert record["frames"]["head"]["timestamp"] == 1.01
    assert record["frames"]["wrist"]["joints"] == {"slide": 0.0}
    assert len(record["frames"]["head"]["sha256"]) == 64
    assert list(tmp_path.glob("*/robot.urdf"))
    pipeline.close()
    assert not pipeline.memory.cells


def test_camera_info_change_invalidates_memory_and_missing_info_blocks():
    d = document()
    d["cameras"]["head"]["require_camera_info"] = True
    pipeline = HardwarePerception(CalibrationProfile(d, XML), clock=lambda: 0.1)
    pipeline.step()
    assert pipeline.snapshot()["state"] == "blocked"
    camera = d["cameras"]["head"]
    info = CameraInfoSample(
        "head", camera["frame_id"], 192, 96, camera["intrinsics"], camera["distortion"], "plumb_bob"
    )
    pipeline.calibration_info(info)
    assert pipeline.verified_info == {"head"}
    pipeline.calibration_info(replace(info, width=640))
    pipeline.step()
    assert pipeline.snapshot()["state"] == "fault"
    assert not pipeline.memory.cells


def test_queues_are_bounded_and_no_depth_does_not_invent_geometry():
    sync = Synchronizer(profile())
    for i in range(300):
        sync.add(joint(i * 0.001), 0)
        sync.add(frame("head", i * 0.001), 0)
    assert len(sync.joints) == 256 and len(sync.frames["head"]) == 16
    cv = pytest.importorskip("cv2")
    image = cv.imencode(".png", np.zeros((96, 192, 3), np.uint8))[1].tobytes()
    pipeline = HardwarePerception(profile(), clock=lambda: 0.1)
    pipeline.ingest(joint(1))
    pipeline.ingest(joint(1.02))
    for name in ("head", "wrist"):
        pipeline.ingest(frame(name, 1.01, image))
    pipeline.step()
    assert pipeline.snapshot()["state"] == "blocked"
    assert not pipeline.memory.cells


def test_clock_reset_after_consumed_camera_frames_latches_fault():
    sync = Synchronizer(profile())
    ready(sync, t=10.0)
    assert sync.next(0.04)
    assert not sync.frames["head"]
    sync.add(frame("head", 1.0), 0.05)
    assert sync.fault


def test_scene_export_is_fresh_isolated_and_invalidated_on_fault():
    now = [0.1]
    pipeline = HardwarePerception(profile(), clock=lambda: now[0])
    left, right = textured_frames()
    for sample in (joint(1), joint(1.02), frame("head", 1.01, left), frame("wrist", 1.01, right)):
        pipeline.ingest(sample)
    pipeline.step()
    scene = pipeline.scene_snapshot()
    assert scene["world_frame"] == "lab" and scene["points"]
    scene["points"].clear()
    assert pipeline.scene_snapshot()["points"]
    now[0] = 1.0
    with pytest.raises(ValueError, match="fresh"):
        pipeline.scene_snapshot()
    pipeline.invalidate("calibration changed")
    assert pipeline.snapshot()["state"] == "fault"


def test_rotation_uncertainty_includes_exposure_time_joint_motion():
    d = document()
    p = CalibrationProfile(d, XML.replace("prismatic", "revolute"))
    assert p.rotation_uncertainty("wrist", 0.02) > p.rotation_uncertainty("head", 0.02)
    assert p.rotation_uncertainty("wrist", 0.04) > p.rotation_uncertainty("wrist", 0.02)


def test_profile_rejects_unknown_fields_and_invalid_dates():
    d = document()
    d["max_skew_ms"] = 10
    with pytest.raises(ValueError, match="Unknown"):
        CalibrationProfile(d, XML)
    d = document()
    d["calibrated_at"] = "2026-11-09"
    with pytest.raises(ValueError, match="timezone"):
        CalibrationProfile(d, XML)


def test_measured_camera_offsets_align_acquisition_times():
    d = document()
    d["cameras"]["wrist"]["time_offset_s"] = 0.002
    sync = Synchronizer(CalibrationProfile(d, XML))
    sync.add(joint(1), 0)
    sync.add(joint(1.02), 0.01)
    sync.add(frame("head", 1.01), 0.02)
    sync.add(frame("wrist", 1.008), 0.03)
    assert sync.next(0.04).skew_s == pytest.approx(0, abs=1e-12)


def test_discontinuous_encoder_motion_is_not_interpolated():
    sync = Synchronizer(profile())
    sync.add(joint(1, 0), 0)
    sync.add(joint(1.02, 0.1), 0.01)
    with pytest.raises(ValueError, match="velocity"):
        sync.interpolate(1.01)


def test_studio_fuses_only_profiled_hardware_and_faults_on_calibration_change(tmp_path):
    import time

    from fastapi.testclient import TestClient

    from fixtures.api import create_app

    class Backend:
        def start(self, sink):
            self.sink = sink

        def close(self):
            self.closed = True

    backend = Backend()
    app = create_app(
        project_dir=tmp_path,
        hardware_backends={"robot": lambda: backend},
        hardware_profiles={"robot": profile()},
    )
    with TestClient(app) as client:
        assert client.get("/api/hardware/scene").status_code == 409
        client.post("/api/hardware/connect", json={"backend": "robot"}).raise_for_status()
        left, right = textured_frames()
        backend.sink.joints(joint(1))
        backend.sink.camera(frame("head", 1.01, left))
        backend.sink.camera(frame("wrist", 1.01, right))
        # Future encoder sample enables interpolation, not extrapolation.
        backend.sink.joints(joint(1.02))
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if client.get("/api/hardware").json()["perception"].get("bundles", 0):
                break
            time.sleep(0.01)
        scene = client.get("/api/hardware/scene")
        assert scene.status_code == 200 and scene.json()["points"]
        assert scene.json()["has_current_depth"]
        camera = document()["cameras"]["head"]
        backend.sink.calibration(
            CameraInfoSample(
                "head",
                "wrong_optical",
                192,
                96,
                camera["intrinsics"],
                camera["distortion"],
                "plumb_bob",
            )
        )
        assert client.get("/api/hardware/scene").status_code == 409
        assert client.get("/api/hardware").json()["perception"]["state"] == "fault"
        client.post("/api/hardware/disconnect", json={}).raise_for_status()
        assert client.get("/api/hardware/scene").status_code == 409
    assert backend.closed
    assert list((tmp_path / "hardware").glob("*/manifest.json"))
