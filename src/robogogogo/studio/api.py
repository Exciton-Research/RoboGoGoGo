"""FastAPI transport and WebSocket lifecycle for a local VLA workspace."""

import asyncio
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import UUID

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.background import BackgroundTask
from starlette.middleware.trustedhost import TrustedHostMiddleware

from robogogogo.robot import HardwareService

from ..simulation.structural_lab import StructuralLab
from .schemas import (
    CameraObservation,
    Command,
    ImportRobot,
    NewEpisode,
    PlanTask,
    PolicyAction,
    SaveProject,
    StereoRequest,
    TaskStep,
)
from .workspace import Studio


class StructuralReset(BaseModel):
    material: Literal["soft_foam", "soft_elastomer", "fragile_shell"] = "soft_foam"
    recipe: Literal["bend", "tear", "crush"] = "bend"


class HardwareConnection(BaseModel):
    backend: str


def create_app(
    *,
    project_dir=None,
    static_dir=None,
    initial_scene="arm",
    hardware_backends=None,
    hardware_profiles=None,
):
    directory = Path(project_dir or Path.cwd() / "runs" / "studio")

    @asynccontextmanager
    async def lifespan(app):
        app.state.studio = await asyncio.to_thread(Studio, directory, scene=initial_scene)
        app.state.clients = 0
        app.state.hardware = HardwareService(
            hardware_backends or {},
            perception_profiles=hardware_profiles,
            audit_dir=directory / "hardware",
        )

        async def clock():
            while True:
                start = time.perf_counter()
                await asyncio.to_thread(app.state.studio.tick)
                studio = app.state.studio
                period = float(studio.session.model.dt) * studio.session.substeps * 2 / studio.speed
                await asyncio.sleep(max(0.001, period - (time.perf_counter() - start)))

        task = asyncio.create_task(clock())
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            await asyncio.to_thread(app.state.hardware.close)

    app = FastAPI(title="robogogogo Studio API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"]
    )

    def allowed_origin(origin, host):
        return origin is None or origin in (
            f"http://{host}",
            "http://127.0.0.1:3000",
            "http://localhost:3000",
        )

    @app.middleware("http")
    async def local_origin(request: Request, call_next):
        if not allowed_origin(request.headers.get("origin"), request.headers.get("host")):
            return JSONResponse(
                {"detail": "This local API does not accept requests from external websites"},
                status_code=403,
            )
        return await call_next(request)

    @app.exception_handler(ValueError)
    async def invalid_request(_, error):
        return JSONResponse({"detail": str(error)}, status_code=422)

    @app.exception_handler(FileNotFoundError)
    async def missing_file(_, error):
        return JSONResponse({"detail": "Project not found"}, status_code=404)

    structural_lock = asyncio.Lock()
    structural_lab = None

    @app.post("/api/structure/reset")
    async def structure_reset(body: StructuralReset):
        nonlocal structural_lab
        async with structural_lock:
            structural_lab = await asyncio.to_thread(StructuralLab, body.material, body.recipe)
            return structural_lab.snapshot()

    @app.post("/api/structure/step")
    async def structure_step():
        async with structural_lock:
            if structural_lab is None:
                raise ValueError("Reset a structural specimen first")
            return await asyncio.to_thread(structural_lab.step)

    @app.get("/api/structure/export")
    async def structure_export():
        async with structural_lock:
            if structural_lab is None:
                raise ValueError("Reset a structural specimen first")
            return JSONResponse(
                structural_lab.export(),
                headers={"Content-Disposition": 'attachment; filename="robogogogo-structure.json"'},
            )

    @app.get("/api/hardware")
    async def hardware_status():
        return app.state.hardware.snapshot()

    @app.post("/api/hardware/connect")
    async def hardware_connect(body: HardwareConnection):
        return await asyncio.to_thread(app.state.hardware.connect, body.backend)

    @app.post("/api/hardware/disconnect")
    async def hardware_disconnect():
        await asyncio.to_thread(app.state.hardware.close)
        return app.state.hardware.snapshot()

    @app.get("/api/hardware/scene")
    async def hardware_scene():
        try:
            return JSONResponse(app.state.hardware.scene(), headers={"Cache-Control": "no-store"})
        except ValueError as error:
            return JSONResponse({"detail": str(error)}, status_code=409)

    @app.get("/api/hardware/cameras/{name}")
    async def hardware_camera(name: str):
        try:
            frame = app.state.hardware.image(name)
        except FileNotFoundError:
            return JSONResponse({"detail": "Camera frame unavailable"}, status_code=404)
        return Response(
            frame.data,
            media_type=frame.media_type,
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )

    @app.get("/api/snapshot")
    async def snapshot():
        return await asyncio.to_thread(app.state.studio.snapshot)

    @app.post("/api/command")
    async def command(body: Command):
        return await asyncio.to_thread(app.state.studio.command, body)

    @app.post("/api/robot")
    async def import_robot(body: ImportRobot):
        return await asyncio.to_thread(
            app.state.studio.import_robot,
            body.xml,
            [arm.model_dump() for arm in body.arms] if body.arms else None,
        )

    @app.get("/api/projects")
    async def projects():
        return await asyncio.to_thread(app.state.studio.projects)

    @app.post("/api/projects")
    async def save(body: SaveProject):
        return await asyncio.to_thread(app.state.studio.save, body)

    @app.post("/api/projects/{identifier}/load")
    async def load(identifier: str):
        return await asyncio.to_thread(app.state.studio.load, identifier)

    @app.get("/api/export")
    async def export():
        path = await asyncio.to_thread(app.state.studio.export)
        return FileResponse(
            path,
            media_type="application/octet-stream",
            filename=f"robogogogo-{datetime.now():%Y%m%d-%H%M%S}.npz",
            background=BackgroundTask(path.unlink, missing_ok=True),
        )

    @app.post("/api/episodes/reset")
    @app.post("/api/vla/reset")
    async def new_episode(body: NewEpisode):
        return await asyncio.to_thread(app.state.studio.episode_service.start_episode, body)

    @app.post("/api/episodes/observe")
    @app.post("/api/vla/observe")
    async def observe(body: CameraObservation):
        return await asyncio.to_thread(app.state.studio.episode_service.capture_observation, body)

    @app.post("/api/episodes/act")
    @app.post("/api/vla/act")
    async def act(body: PolicyAction):
        return await asyncio.to_thread(app.state.studio.episode_service.policy_action, body)

    @app.post("/api/tasks/plan")
    async def plan_task(body: PlanTask):
        return await asyncio.to_thread(app.state.studio.task_service.plan, body)

    @app.post("/api/tasks/step")
    async def task_step(body: TaskStep):
        return await asyncio.to_thread(app.state.studio.task_service.step, body.observation)

    @app.post("/api/tasks/pause")
    async def pause_task():
        return await asyncio.to_thread(app.state.studio.task_service.pause)

    @app.post("/api/tasks/resume")
    async def resume_task():
        return await asyncio.to_thread(app.state.studio.task_service.resume)

    @app.post("/api/perception/stereo")
    async def stereo(body: StereoRequest):
        from .perception import analyze_stereo

        return await asyncio.to_thread(analyze_stereo, app.state.studio, body)

    @app.get("/api/control/rule-based-action")
    async def rule_action():
        return await asyncio.to_thread(app.state.studio.episode_service.rule_action)

    @app.get("/api/vla/reference-action")
    async def reference_action():
        return await asyncio.to_thread(app.state.studio.episode_service.reference_action)

    @app.get("/api/episodes/episodes")
    @app.get("/api/vla/episodes")
    async def episodes():
        return await asyncio.to_thread(app.state.studio.episode_service.episodes)

    @app.post("/api/episodes/save")
    @app.post("/api/vla/save")
    async def save_episode():
        return await asyncio.to_thread(app.state.studio.episode_service.save_episode)

    @app.get("/api/episodes/episodes/{identifier}")
    @app.get("/api/vla/episodes/{identifier}")
    async def download_episode(identifier: str):
        try:
            identifier = str(UUID(identifier))
        except ValueError as error:
            raise ValueError("Invalid episode ID") from error
        path = directory / "episodes" / f"{identifier}.zip"
        if not path.exists():
            raise FileNotFoundError("Episode not found")
        return FileResponse(
            path, media_type="application/zip", filename=f"robogogogo-episode-{identifier[:8]}.zip"
        )

    @app.websocket("/api/ws")
    async def stream(websocket: WebSocket):
        if not allowed_origin(websocket.headers.get("origin"), websocket.headers.get("host")):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        app.state.clients += 1

        async def receive():
            while True:
                await websocket.receive_text()

        receiver = asyncio.create_task(receive())
        try:
            while not receiver.done():
                await websocket.send_json(await asyncio.to_thread(app.state.studio.snapshot))
                await asyncio.sleep(0.04)
        except (WebSocketDisconnect, RuntimeError, OSError):
            pass
        finally:
            receiver.cancel()
            try:
                await receiver
            except (asyncio.CancelledError, WebSocketDisconnect):
                pass
            app.state.clients -= 1
            if app.state.clients == 0:
                await asyncio.to_thread(app.state.studio.command, Command(op="pause"))

    static = Path(static_dir) if static_dir else Path(__file__).parents[1] / "web"
    if static.is_dir():
        app.mount("/", StaticFiles(directory=static, html=True), name="studio")
    else:

        @app.get("/")
        async def not_built():
            return JSONResponse(
                {"detail": "Studio assets are not built. Run npm ci && npm run build in studio/."},
                status_code=503,
            )

    return app
