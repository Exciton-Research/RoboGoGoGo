import base64
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from robogogogo.hardware import CameraSample, HardwareService, JointSample, SDKBackend
from robogogogo.hardware.ros2 import ROS2Backend, camera_sample, joint_sample
from robogogogo.studio.api import create_app

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a5WQAAAAASUVORK5CYII="
)


class Feed:
    def start(self, sink):
        self.sink = sink
        self.closed = False

    def close(self):
        self.closed = True


def joint(timestamp=1):
    return JointSample(timestamp, ("left_shoulder", "right_shoulder"), (0.1, -0.2))


def camera(timestamp=1, name="head_left"):
    return CameraSample(name, timestamp, PNG, "image/png", "head_left_optical")


def test_stale_duplicate_and_out_of_order_samples_do_not_look_live():
    now = [0.0]
    feed = Feed()
    service = HardwareService({"robot": lambda: feed}, clock=lambda: now[0])
    assert service.connect("robot")["status"] == "waiting"
    feed.sink.joints(joint())
    feed.sink.camera(camera())
    snapshot = service.snapshot()
    assert snapshot["status"] == "streaming" and snapshot["read_only"]
    assert snapshot["joints"]["names"] == ("left_shoulder", "right_shoulder")
    assert snapshot["joints"]["velocity"] == ()
    now[0] = 3
    feed.sink.joints(joint())
    feed.sink.camera(camera(0.5))
    assert service.snapshot()["status"] == "stale"
    feed.sink.joints(joint(2))
    assert not service.snapshot()["joints"]["stale"]
    assert service.snapshot()["cameras"][0]["stale"]
    service.close()
    assert feed.closed and service.snapshot()["joints"] is None
    feed.sink.joints(joint(3))
    assert service.snapshot()["joints"] is None


@pytest.mark.parametrize(
    "sample",
    [
        replace(joint(), position=(float("nan"), 0.2)),
        replace(joint(), names=("same", "same")),
        replace(joint(), velocity=(1.0,)),
        replace(camera(), data=b"not an image"),
        replace(camera(), timestamp=-1),
    ],
)
def test_invalid_telemetry_faults_without_replacing_valid_data(sample):
    feed = Feed()
    service = HardwareService({"robot": lambda: feed})
    service.connect("robot")
    feed.sink.joints(joint())
    feed.sink.camera(sample)
    assert service.snapshot()["status"] == "fault"
    assert service.snapshot()["joints"]["position"] == (0.1, -0.2)


def test_reconnect_discards_old_callbacks_and_failed_start_cleans_up():
    feeds = []

    def factory():
        feed = Feed()
        feeds.append(feed)
        return feed

    service = HardwareService({"robot": factory})
    service.connect("robot")
    service.connect("robot")
    feeds[0].sink.camera(camera())
    assert not service.snapshot()["cameras"] and feeds[0].closed
    feeds[1].sink.camera(camera())
    assert len(service.snapshot()["cameras"]) == 1

    class Broken(Feed):
        def start(self, sink):
            super().start(sink)
            raise RuntimeError("connection refused")

    broken = Broken()
    other = HardwareService({"bad": lambda: broken})
    with pytest.raises(ValueError, match="connection refused"):
        other.connect("bad")
    assert broken.closed and other.snapshot()["status"] == "fault"
    service.close()


def test_sdk_reads_only_and_stops():
    received = threading.Event()

    class Reader:
        def read(self, timeout_s):
            assert timeout_s > 0
            received.set()
            return joint(), (camera(),)

        def close(self):
            self.closed = True

    reader = Reader()
    service = HardwareService({"sdk": lambda: SDKBackend(reader)})
    service.connect("sdk")
    assert received.wait(1)
    service.close()
    assert reader.closed
    assert service.snapshot()["status"] == "disconnected"


def test_ros_converters_keep_names_timestamps_and_missing_velocity():
    header = SimpleNamespace(stamp=SimpleNamespace(sec=10, nanosec=500000000), frame_id="base")
    sample = joint_sample(
        SimpleNamespace(header=header, name=["b", "a"], position=[0.2, 0.1], velocity=[], effort=[])
    )
    assert sample.timestamp == 10.5 and sample.names == ("b", "a") and sample.velocity == ()
    image = camera_sample(
        "wrist", SimpleNamespace(header=header, format="rgb8; png compressed", data=PNG)
    )
    assert image.media_type == "image/png" and image.frame_id == "base"
    with pytest.raises(ValueError, match="compressedDepth"):
        camera_sample(
            "wrist", SimpleNamespace(header=header, format="16UC1; compressedDepth", data=PNG)
        )
    with pytest.raises(ValueError):
        ROS2Backend(cameras={"bad/name": "/camera"})


def test_hardware_api_is_read_only_and_separate_from_simulation(tmp_path):
    feed = Feed()
    app = create_app(project_dir=tmp_path, hardware_backends={"lab_robot": lambda: feed})
    with TestClient(app) as client:
        before = client.get("/api/snapshot").json()
        assert client.get("/api/hardware").json()["backends"] == ["lab_robot"]
        assert client.post("/api/hardware/connect", json={"backend": "unknown"}).status_code == 422
        assert (
            client.post("/api/hardware/connect", json={"backend": "lab_robot"}).status_code == 200
        )
        feed.sink.joints(joint())
        feed.sink.camera(camera())
        feed.sink.camera(camera(name="wrist_right"))
        state = client.get("/api/hardware").json()
        assert len(state["cameras"]) == 2 and state["joints"]["position"] == [0.1, -0.2]
        response = client.get("/api/hardware/cameras/head_left")
        assert response.content == PNG and response.headers["cache-control"] == "no-store"
        assert client.post("/api/hardware/act", json={}).status_code in (404, 405)
        assert (
            client.post(
                "/api/hardware/connect",
                json={"backend": "lab_robot"},
                headers={"origin": "https://untrusted.example"},
            ).status_code
            == 403
        )
        after = client.get("/api/snapshot").json()
        assert before["model"] == after["model"]
        client.post("/api/hardware/disconnect", json={})
        assert client.get("/api/hardware/cameras/head_left").status_code == 404
    assert feed.closed


def test_cleanup_failure_disconnects_telemetry_and_reports_fault():
    class BrokenClose(Feed):
        def close(self):
            raise RuntimeError("reader shutdown timed out")

    feed = BrokenClose()
    service = HardwareService({"robot": lambda: feed})
    service.connect("robot")
    feed.sink.joints(joint())
    with pytest.raises(ValueError, match="cleanup failed"):
        service.close()
    feed.sink.joints(joint(2))
    snapshot = service.snapshot()
    assert snapshot["status"] == "fault" and snapshot["joints"] is None


def test_ros_backend_subscribes_and_cleans_up_owned_context(monkeypatch):
    import sys

    nodes, executors, initialized = [], [], []

    class Context:
        def __init__(self):
            self.live = True

        def ok(self):
            return self.live

        def try_shutdown(self):
            self.live = False

    class Node:
        def __init__(self, name, **kwargs):
            assert not kwargs["enable_rosout"] and not kwargs["start_parameter_services"]
            self.subscriptions = []
            self.destroyed = False
            nodes.append(self)

        def create_subscription(self, message_type, topic, callback, qos):
            self.subscriptions.append((topic, callback))

        def destroy_node(self):
            self.destroyed = True

    class Executor:
        def __init__(self, context):
            self.closed = False
            executors.append(self)

        def add_node(self, node):
            self.node = node

        def spin_once(self, timeout_sec):
            threading.Event().wait(0.001)

        def shutdown(self, timeout_sec):
            self.closed = True

    modules = {
        "rclpy": SimpleNamespace(init=lambda **kwargs: initialized.append(kwargs)),
        "rclpy.context": SimpleNamespace(Context=Context),
        "rclpy.node": SimpleNamespace(Node=Node),
        "rclpy.executors": SimpleNamespace(SingleThreadedExecutor=Executor),
        "rclpy.qos": SimpleNamespace(qos_profile_sensor_data=object()),
        "sensor_msgs": SimpleNamespace(),
        "sensor_msgs.msg": SimpleNamespace(JointState=object(), CompressedImage=object()),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    service = HardwareService(
        {"ros2": lambda: ROS2Backend(cameras={"head": "/head/image/compressed"}, domain_id=5)}
    )
    service.connect("ros2")
    assert initialized[0]["domain_id"] == 5
    callbacks = dict(nodes[0].subscriptions)
    assert set(callbacks) == {"/joint_states", "/head/image/compressed"}
    header = SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=0), frame_id="robot")
    callbacks["/joint_states"](
        SimpleNamespace(header=header, name=["joint1"], position=[0.1], velocity=[], effort=[])
    )
    callbacks["/head/image/compressed"](SimpleNamespace(header=header, format="png", data=PNG))
    assert service.snapshot()["status"] == "streaming"
    assert service.snapshot()["cameras"][0]["name"] == "head"
    service.close()
    assert nodes[0].destroyed and executors[0].closed and not initialized[0]["context"].ok()


def test_camera_budget_is_bounded():
    feed = Feed()
    service = HardwareService({"robot": lambda: feed})
    service.connect("robot")
    for index in range(9):
        feed.sink.camera(camera(name=f"camera_{index}"))
    assert len(service.snapshot()["cameras"]) == 8
    assert service.snapshot()["status"] == "fault"
    service.close()
