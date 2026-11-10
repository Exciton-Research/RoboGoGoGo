"""Local Studio launcher. Domain services live in robogogogo.studio."""

from pathlib import Path

from .studio.api import create_app
from .studio.schemas import Command
from .studio.workspace import Studio

__all__ = ["create_app", "Studio", "Command", "main"]


def main():
    import argparse

    import jax
    import uvicorn

    parser = argparse.ArgumentParser(
        description="robogogogo Studio — local TypeScript + Python/JAX workbench"
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--projects", type=Path, default=Path.cwd() / "runs" / "studio")
    parser.add_argument(
        "--ros-joints", help="Register a read-only ROS 2 backend with this JointState topic"
    )
    parser.add_argument(
        "--ros-camera",
        action="append",
        default=[],
        metavar="NAME=TOPIC",
        help="CompressedImage topic; repeat for head and wrist cameras",
    )
    parser.add_argument("--ros-domain-id", type=int, default=None)
    args = parser.parse_args()
    backends = {}
    if args.ros_camera and not args.ros_joints:
        parser.error("--ros-camera requires --ros-joints")
    if args.ros_joints:
        from .hardware.ros2 import ROS2Backend

        cameras = {}
        for entry in args.ros_camera:
            if "=" not in entry:
                parser.error("Camera format is NAME=/topic/image/compressed")
            name, topic = entry.split("=", 1)
            if name in cameras:
                parser.error("Camera names must be unique")
            cameras[name] = topic
        try:
            ROS2Backend(joint_topic=args.ros_joints, cameras=cameras, domain_id=args.ros_domain_id)
        except ValueError as error:
            parser.error(str(error))
        backends["ros2"] = lambda: ROS2Backend(
            joint_topic=args.ros_joints, cameras=cameras, domain_id=args.ros_domain_id
        )
    if not 1024 <= args.port <= 65535:
        parser.error("port must be between 1024 and 65535")
    jax.config.update("jax_enable_x64", True)
    print(f"robogogogo Studio → http://127.0.0.1:{args.port} (warming up JAX)", flush=True)
    uvicorn.run(
        create_app(project_dir=args.projects, hardware_backends=backends),
        host="127.0.0.1",
        port=args.port,
    )


if __name__ == "__main__":
    main()
