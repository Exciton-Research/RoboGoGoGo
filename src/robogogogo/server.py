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
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("port must be between 1024 and 65535")
    jax.config.update("jax_enable_x64", True)
    print(f"robogogogo Studio → http://127.0.0.1:{args.port} (warming up JAX)", flush=True)
    uvicorn.run(create_app(project_dir=args.projects), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
