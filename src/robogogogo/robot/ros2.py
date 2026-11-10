"""Optional ROS 2 subscriptions only; no actuator publishers or command services."""

import re
import threading

from robogogogo.robot.contracts import (
    CameraInfoSample,
    CameraSample,
    JointSample,
    TelemetrySink,
    TransformSample,
)


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
        "ros",
    )


def camera_sample(name, message):
    data = bytes(message.data)
    if "compressedDepth" in message.format:
        raise ValueError("Configure RGB compressed topics, not compressedDepth")
    media_type = "image/png" if data.startswith(b"\x89PNG\r\n\x1a\n") else "image/jpeg"
    return CameraSample(
        name, timestamp(message.header), data, media_type, message.header.frame_id, "ros"
    )


class ROS2Backend:
    def __init__(
        self,
        *,
        joint_topic="/joint_states",
        cameras=None,
        domain_id=None,
        camera_info_topics=None,
        localization_chain=None,
        tf_topic="/tf",
        tf_static_topic="/tf_static",
    ):
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
        self.info_topics = dict(camera_info_topics or {})
        if not set(self.info_topics).issubset(self.cameras) or any(
            not t.startswith("/") for t in self.info_topics.values()
        ):
            raise ValueError("CameraInfo topics must correspond to configured cameras")
        self.localization_chain = tuple(localization_chain or ())
        if self.localization_chain and (
            not 2 <= len(self.localization_chain) <= 8
            or len(set(self.localization_chain)) != len(self.localization_chain)
        ):
            raise ValueError("ROS localization requires an explicit acyclic world-to-base TF chain")
        if not tf_topic.startswith("/") or not tf_static_topic.startswith("/"):
            raise ValueError("TF topics must be absolute names")
        self.tf_topic, self.tf_static_topic = tf_topic, tf_static_topic
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
        if self.info_topics:
            from sensor_msgs.msg import CameraInfo

            def info(name, message):
                try:
                    if (
                        message.binning_x > 1
                        or message.binning_y > 1
                        or message.roi.width
                        or message.roi.height
                    ):
                        raise ValueError(
                            "Binned/cropped CameraInfo requires an explicitly adjusted calibration"
                        )
                    if sink.calibration:
                        sink.calibration(
                            CameraInfoSample(
                                name,
                                message.header.frame_id,
                                message.width,
                                message.height,
                                tuple(tuple(message.k[i : i + 3]) for i in range(0, 9, 3)),
                                tuple(message.d),
                                message.distortion_model,
                            )
                        )
                except Exception as error:
                    sink.error(f"Invalid ROS CameraInfo: {error}")

            for name, topic in self.info_topics.items():
                self.node.create_subscription(
                    CameraInfo,
                    topic,
                    lambda message, name=name: info(name, message),
                    qos_profile_sensor_data,
                )
        if self.localization_chain:
            from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
            from tf2_msgs.msg import TFMessage

            def transforms(message, static):
                try:
                    if len(message.transforms) > 1024:
                        raise ValueError("TF message exceeds telemetry transform budget")
                    for item in message.transforms:
                        if item.child_frame_id not in self.localization_chain[1:]:
                            continue
                        t, q = item.transform.translation, item.transform.rotation
                        if sink.localization:
                            sink.localization(
                                TransformSample(
                                    timestamp(item.header),
                                    item.header.frame_id,
                                    item.child_frame_id,
                                    (t.x, t.y, t.z),
                                    (q.x, q.y, q.z, q.w),
                                    "ros",
                                    static,
                                )
                            )
                except Exception as error:
                    sink.error(f"Invalid ROS localization TF: {error}")

            self.node.create_subscription(
                TFMessage,
                self.tf_topic,
                lambda message: transforms(message, False),
                qos_profile_sensor_data,
            )
            # Static transforms must arrive for late subscribers as well.
            static_qos = QoSProfile(
                depth=100,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                reliability=ReliabilityPolicy.RELIABLE,
            )
            self.node.create_subscription(
                TFMessage,
                self.tf_static_topic,
                lambda message: transforms(message, True),
                static_qos,
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
