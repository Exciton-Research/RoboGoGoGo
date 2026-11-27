"""Launch just the grasp-and-place demo, without warming up the legacy JAX Studio."""

import argparse
from pathlib import Path


def create_app(cache, static_dir=None, recordings=None):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from fastapi.staticfiles import StaticFiles
    from starlette.middleware.gzip import GZipMiddleware
    from starlette.middleware.trustedhost import TrustedHostMiddleware

    from .api import create_demo_router

    app = FastAPI(title="Panda grasp and place")
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]"])
    app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=3)
    app.include_router(create_demo_router(cache, recordings=recordings))

    @app.middleware("http")
    async def local_origin(request: Request, call_next):
        host = request.headers.get("host")
        origin = request.headers.get("origin")
        if origin is not None and origin not in (
            f"http://{host}",
            "http://127.0.0.1:3000",
            "http://localhost:3000",
        ):
            return JSONResponse({"detail": "Local requests only"}, status_code=403)
        return await call_next(request)

    static = Path(static_dir) if static_dir else Path(__file__).parents[1] / "web"
    if static.is_dir():
        app.mount("/", StaticFiles(directory=static, html=True), name="demo")
    else:

        @app.get("/")
        async def not_built():
            return JSONResponse(
                {"detail": "Build the demo UI with npm ci && npm run build in studio/."},
                status_code=503,
            )

    return app


def main():
    parser = argparse.ArgumentParser(description="Panda grasp-and-place demo")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--assets", type=Path, default=Path.cwd() / "runs" / "demo" / "assets")
    parser.add_argument(
        "--recordings", type=Path, default=Path.cwd() / "runs" / "demo" / "episodes"
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Download the Panda model and two YCB objects, then exit",
    )
    args = parser.parse_args()
    if args.setup:
        from .assets import download
        from .objects import CATALOG, download_object

        download(args.assets)
        for identifier in CATALOG:
            print(f"Preparing {identifier}…", flush=True)
            download_object(args.assets, identifier)
        return
    if not 1024 <= args.port <= 65535:
        parser.error("port must be between 1024 and 65535")
    import uvicorn

    uvicorn.run(
        create_app(args.assets, recordings=args.recordings),
        host="127.0.0.1",
        port=args.port,
        access_log=False,
    )


if __name__ == "__main__":
    main()
