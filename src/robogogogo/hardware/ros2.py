"""Optional ROS 2 subscriptions only; no actuator publishers or command services."""

import re
import threading

from .contracts import CameraSample, JointSample, TelemetrySink


def timestamp(header):
    return float(header.stamp.sec) + float(header.stamp.nanosec) * 1e-9


def joint_sample(message):
    return JointSample(
        timestamp(message.header),
        tuple(message.name),
        tuple(message.position),
        tuple(message.velocity),
        tuple(message.effort),
        message.header.frame_id,
    )


def camera_sample(name, message):
    data = bytes(message.data)
    if "compressedDepth" in message.format:
        raise ValueError("Configure RGB compressed topics, not compressedDepth")
    media_type = "image/png" if data.startswith(b"\x89PNG\r\n\x1a\n") else "image/jpeg"
    return CameraSample(name, timestamp(message.header), data, media_type, message.header.frame_id)


class ROS2Backend:
    def __init__(self, *, joint_topic="/joint_states", cameras=None, domain_id=None):
        self.joint_topic, self.cameras = joint_topic, dict(cameras or {})
        if len(self.cameras) > 8 or any(
            not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", n) for n in self.cameras
        ):
            raise ValueError("Configure up to eight named ROS cameras")
        if any(
            not isinstance(t, str) or not t.startswith("/") or len(t) > 256
            for t in (joint_topic, *self.cameras.values())
        ):
            raise ValueError("ROS topics must be absolute names")
        if domain_id is not None and (not isinstance(domain_id, int) or not 0 <= domain_id <= 232):
            raise ValueError("ROS domain ID must be between 0 and 232")
        self.domain_id = domain_id
        self.context = self.node = self.executor = self.thread = None
        self.stop = threading.Event()

    def start(self, sink: TelemetrySink):
        if self.context is not None:
            raise ValueError("Create a new ROS adapter for each connection")
        try:
            import rclpy
            from rclpy.context import Context
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.node import Node
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import CompressedImage, JointState
        except ImportError as error:
            raise ValueError(
                "ROS 2 is unavailable. Source your ROS installation and use a Python environment with rclpy and sensor_msgs."
            ) from error
        self.context = Context()
        rclpy.init(args=[], context=self.context, domain_id=self.domain_id)
        self.node = Node(
            "robogogogo_telemetry",
            context=self.context,
            enable_rosout=False,
            start_parameter_services=False,
        )

        def joints(message):
            try:
                sink.joints(joint_sample(message))
            except Exception as error:
                sink.error(f"Invalid ROS joint sample: {error}")

        def image(name, message):
            try:
                sink.camera(camera_sample(name, message))
            except Exception as error:
                sink.error(f"Invalid ROS camera sample: {error}")

        self.node.create_subscription(JointState, self.joint_topic, joints, qos_profile_sensor_data)
        for name, topic in self.cameras.items():
            self.node.create_subscription(
                CompressedImage,
                topic,
                lambda message, name=name: image(name, message),
                qos_profile_sensor_data,
            )
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)

        def spin():
            try:
                while not self.stop.is_set() and self.context.ok():
                    self.executor.spin_once(timeout_sec=0.1)
            except Exception as error:
                if not self.stop.is_set():
                    sink.error(f"ROS telemetry stopped: {error}")

        self.thread = threading.Thread(target=spin, daemon=True, name="robot-ros-telemetry")
        self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
            if self.thread.is_alive():
                raise RuntimeError("ROS telemetry executor did not stop")
        if self.executor is not None:
            self.executor.shutdown(timeout_sec=2)
        if self.node is not None:
            self.node.destroy_node()
        if self.context is not None:
            self.context.try_shutdown()
