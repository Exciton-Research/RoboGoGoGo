from dataclasses import replace

import numpy as np
import pytest
from test_hardware_perception import XML, document, frame, joint, textured_frames

from robogogogo.hardware import TransformSample
from robogogogo.hardware.calibration import CalibrationProfile, axis_rotation
from robogogogo.hardware.localization import LocalizationBuffer
from robogogogo.hardware.perception import HardwarePerception
from robogogogo.hardware.synchronization import Synchronizer


def moving_profile():
    d = document()
    d["fixed_base"] = False
    del d["world_from_base"]
    d["world_frame"] = "map"
    d["localization"] = dict(
        tf_chain=["map", "odom", "base"],
        max_gap_s=0.04,
        max_age_s=0.3,
        translation_uncertainty_m=0.0002,
        rotation_uncertainty_rad=0.00005,
        max_linear_speed_m_s=0.2,
        max_angular_speed_rad_s=0.2,
        jump_translation_m=0.2,
        jump_rotation_rad=0.2,
    )
    return CalibrationProfile(d, XML)


def pose(t, x=0.0, yaw=0.0, *, parent="odom", child="base", static=False, epoch=""):
    return TransformSample(
        t,
        parent,
        child,
        (x, 0.0, 0.0),
        (0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)),
        "robot",
        static,
        epoch,
    )


def static_map(x=0.0):
    return pose(0.0, x, parent="map", child="odom", static=True)


def test_chain_interpolates_translation_and_rotation_at_exposure():
    p = moving_profile()
    loc = LocalizationBuffer(p.localization, "robot")
    loc.add(static_map(0.3), 0.0)
    loc.add(pose(1.0), 0.01)
    loc.add(pose(1.02, 0.002, 0.002), 0.02)
    result = loc.at(1.01, 0.03)
    np.testing.assert_allclose(result.world_from_base[:3, 3], [0.301, 0, 0])
    np.testing.assert_allclose(
        result.world_from_base[:3, :3], axis_rotation([0, 0, 1], 0.001), atol=1e-9
    )
    assert result.gap_s == pytest.approx(0.02)
    assert result.edges[0]["static"]
    assert result.interpolation_translation_m > 0
    with pytest.raises(ValueError, match="extrapolation"):
        loc.at(1.03, 0.03)
    with pytest.raises(ValueError, match="Stale"):
        loc.at(1.01, 1.0)


def test_pose_and_joint_motion_both_enter_camera_extrinsics():
    p = moving_profile()
    sync = Synchronizer(p)
    for sample in (
        static_map(),
        pose(1.0),
        pose(1.02, 0.002),
        joint(1, 0),
        joint(1.02, 0.001),
        frame("head", 1.01),
        frame("wrist", 1.012),
    ):
        sync.add(sample, 0.01)
    bundle = sync.next(0.02)
    assert bundle is not None
    head = p.pose("head", bundle.positions["head"], bundle.base_poses["head"].world_from_base)
    wrist = p.pose("wrist", bundle.positions["wrist"], bundle.base_poses["wrist"].world_from_base)
    assert head["position"][0] == pytest.approx(0.001)
    assert wrist["position"][0] == pytest.approx(0.064 + 0.0012 + 0.0006)
    with pytest.raises(ValueError, match="exposure-time localization"):
        p.pose("head", {"slide": 0})


def test_stale_duplicates_clock_reset_parent_conflict_and_quaternion_validation():
    p = moving_profile()
    for bad in (
        replace(pose(1.02), clock_domain="host"),
        replace(pose(1.02), quaternion_xyzw=(0, 0, 0, 0)),
        pose(1.02, parent="new_map"),
        pose(0.0),
    ):
        loc = LocalizationBuffer(p.localization, "robot")
        loc.add(static_map(), 0.0)
        loc.add(pose(2.0), 0.0)
        loc.add(bad, 0.1)
        assert loc.fault
    loc = LocalizationBuffer(p.localization, "robot")
    loc.add(static_map(), 0.0)
    loc.add(pose(1.0), 0.0)
    loc.add(pose(1.0), 1.0)
    assert "Stale" in loc.health(1.0)


def test_gap_rejection_and_static_history_for_late_subscribers():
    p = moving_profile()
    loc = LocalizationBuffer(p.localization, "robot")
    loc.add(static_map(), -100.0)
    loc.add(pose(1.0), 0.1)
    loc.add(pose(1.1), 0.2)
    with pytest.raises(ValueError, match="gap"):
        loc.at(1.05, 0.21)
    assert loc.health(0.21) is None  # static transform does not age out


def test_localization_jump_flushes_sensor_queues_and_excludes_old_epoch_samples():
    sync = Synchronizer(moving_profile())
    for sample in (static_map(), pose(1.0), joint(1), frame("head", 1.01)):
        sync.add(sample, 0.01)
    sync.add(pose(1.02, 1.0), 0.02)
    assert sync.localization_epoch == 1
    assert not sync.joints and not sync.frames["head"]
    sync.add(pose(1.01, 0.5), 0.03)
    assert len(sync.localization.dynamic[("odom", "base")]) == 1
    assert sync.localization.dynamic[("odom", "base")][0][0] == 1.02


def test_static_correction_and_provider_epoch_start_new_memory_epochs():
    loc = LocalizationBuffer(moving_profile().localization, "robot")
    loc.add(static_map(), 0)
    loc.add(pose(1.0), 0.1)
    loc.add(static_map(0.01), 0.2)
    assert loc.epoch == 1 and not loc.dynamic[("odom", "base")]
    loc.add(pose(1.02, epoch="new-localization"), 0.3)
    assert loc.epoch == 2


def test_quaternion_sign_and_small_arrival_reordering_do_not_trigger_resets():
    loc = LocalizationBuffer(moving_profile().localization, "robot")
    loc.add(static_map(), 0.0)
    loc.add(pose(1.0), 0.01)
    loc.add(replace(pose(1.02), quaternion_xyzw=(0, 0, 0, -1)), 0.02)
    loc.add(pose(1.01), 0.03)
    assert loc.epoch == 0 and not loc.fault
    np.testing.assert_allclose(loc.at(1.015, 0.04).world_from_base, np.eye(4))


def observe(pipeline, t, x, left, right):
    for sample in (
        pose(t + 0.01, x),
        joint(t),
        joint(t + 0.02),
        frame("head", t + 0.01, left),
        frame("wrist", t + 0.01, right),
    ):
        pipeline.ingest(sample)
    pipeline.step()


def test_relocalization_invalidates_export_immediately_and_rebuilds_memory(tmp_path):
    pipeline = HardwarePerception(moving_profile(), clock=lambda: 0.1, audit_dir=tmp_path)
    pipeline.ingest(static_map())
    left, right = textured_frames()
    observe(pipeline, 1.0, 0.0, left, right)
    assert pipeline.scene_snapshot()["localization_epoch"] == 0
    pipeline.ingest(pose(1.03, 1.0))
    with pytest.raises(ValueError):
        pipeline.scene_snapshot()
    pipeline.step()
    assert not pipeline.memory.cells
    observe(pipeline, 1.04, 1.0, left, right)
    scene = pipeline.scene_snapshot()
    assert scene["localization_epoch"] == 1 and scene["points"]
    assert min(point[0] for point in scene["points"]) > 0.7
    pipeline.close()


def test_stationary_surface_stays_in_world_coordinates_while_robot_moves():
    cv = pytest.importorskip("cv2")
    texture = np.random.default_rng(32).integers(0, 256, (96, 260, 3), dtype=np.uint8)

    def images(shift):
        return tuple(
            cv.imencode(".png", texture[:, shift + offset : shift + offset + 192])[1].tobytes()
            for offset in (0, 32)
        )

    now = [0.1]
    pipeline = HardwarePerception(moving_profile(), clock=lambda: now[0])
    pipeline.ingest(static_map())
    observe(pipeline, 1.0, 0.0, *images(0))
    original = {
        tuple(cell["color"]): cell["point"].copy() for cell in pipeline.memory.cells.values()
    }
    now[0] = 0.2
    # Plane at z=.24: a 2 cm camera translation produces a 10 pixel shift at fx=120.
    observe(pipeline, 2.0, 0.02, *images(10))
    differences = [
        np.linalg.norm(cell["point"] - original[tuple(cell["color"])])
        for cell in pipeline.memory.cells.values()
        if cell["seen"] > 2 and tuple(cell["color"]) in original
    ]
    assert len(differences) > 20
    assert np.median(differences) < 0.003
    assert pipeline.snapshot()["localization"]["resets"] == 0
    pipeline.close()


def test_stale_localization_blocks_scene_export_even_before_scene_timeout():
    now = [0.1]
    pipeline = HardwarePerception(moving_profile(), clock=lambda: now[0])
    pipeline.ingest(static_map())
    observe(pipeline, 1.0, 0.0, *textured_frames())
    now[0] = 0.41
    with pytest.raises(ValueError):
        pipeline.scene_snapshot()
    assert pipeline.snapshot()["localization"]["reason"].startswith("Stale")
