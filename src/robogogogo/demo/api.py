"""Local transport for the explicitly stepped Panda environment."""

import asyncio
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field

from .pretrained import ModelNotReady


class ResetRequest(BaseModel):
    seed: int = Field(default=0, ge=0, le=1_000_000)


class CommandRequest(ResetRequest):
    command: Literal["run", "pause", "reset"]
    perception: Literal["camera", "simulator", "vlm"] = "simulator"
    query: str | None = Field(default=None, min_length=1, max_length=120)


class CollectRequest(BaseModel):
    images: bool = True


class FindRequest(BaseModel):
    query: str = Field(min_length=1, max_length=120)
    expected_revision: int | None = Field(default=None, ge=0)


class StepRequest(BaseModel):
    action: list[float] = Field(min_length=8, max_length=8)
    expected_revision: int | None = Field(default=None, ge=0)


class ManualRequest(BaseModel):
    delta: list[float] = Field(default=[0, 0, 0], min_length=3, max_length=3)
    opening: float | None = Field(default=None, ge=0, le=0.08)
    expected_revision: int | None = Field(default=None, ge=0)


class RecordRequest(BaseModel):
    command: Literal["start", "save", "discard"]
    images: bool = True


class ReplayRequest(BaseModel):
    command: Literal["play", "pause", "seek", "exit"]
    frame: int | None = Field(default=None, ge=0)


class ObjectRequest(ResetRequest):
    identifier: str
    clutter: bool = False


class TargetRequest(BaseModel):
    identifier: str


def create_demo_router(cache, *, legacy_workspace=False, recordings=None):
    # Construction starts a single worker. No model or rendering work runs on
    # arbitrary FastAPI pool threads, and every mutation passes through its queue.
    session = None
    directory = Path(recordings or Path.cwd() / "runs/demo/episodes")

    def get_session():
        nonlocal session
        if session is None:
            from .session import PandaSession

            session = PandaSession(cache, directory)
        return session

    async def call(method, **kwargs):
        # get_session executes on the event-loop thread, so initialization cannot
        # race between simultaneous first requests.
        try:
            target = get_session()
            return await asyncio.to_thread(target.call, method, **kwargs)
        except ModelNotReady as error:
            raise HTTPException(503, str(error)) from error
        except (ImportError, FileNotFoundError) as error:
            raise HTTPException(
                503, str(error) + ". Install .[studio,demo] and run robogogogo-demo --setup."
            ) from error
        except (ValueError, RuntimeError) as error:
            raise HTTPException(422, str(error)) from error

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            if session is not None:
                await asyncio.to_thread(session.close)

    router = APIRouter(prefix="/api/demo", lifespan=lifespan)

    @router.get("/objects")
    async def objects():
        from .objects import catalog

        return await asyncio.to_thread(catalog, cache)

    @router.post("/objects/select")
    async def select_object(body: ObjectRequest):
        return await call("select_object", **body.model_dump())

    @router.post("/target")
    async def select_target(body: TargetRequest):
        return await call("select_target", identifier=body.identifier)

    @router.post("/objects/{identifier}/download")
    async def download_object(identifier: str):
        from .objects import CATALOG, download_object

        if identifier not in CATALOG:
            raise HTTPException(404, "Unknown library object")

        try:
            return await asyncio.to_thread(download_object, cache, identifier)
        except (OSError, ValueError) as error:
            raise HTTPException(502, f"Object download failed: {error}") from error

    @router.get("/objects/{identifier}/texture")
    async def object_texture(identifier: str):
        from .objects import CATALOG, directory, require_object

        if identifier not in CATALOG:
            raise HTTPException(404, "Unknown library object")

        try:
            await asyncio.to_thread(require_object, cache, identifier)
        except FileNotFoundError as error:
            raise HTTPException(404, str(error)) from error
        return FileResponse(directory(cache, identifier) / "texture.png", media_type="image/png")

    @router.get("/scene")
    async def scene():
        return {**await call("scene"), "legacy_workspace": legacy_workspace}

    @router.get("/state")
    async def state():
        return await call("snapshot")

    @router.get("/observation")
    async def policy_observation(images: bool = False):
        return await call("policy_observation", images=images)

    @router.get("/vision/image")
    async def vision_image(episode_id: str, detection_id: str | None = None):
        return Response(
            await call("vision_image", episode_id=episode_id, detection_id=detection_id),
            media_type="image/png",
            headers={"Cache-Control": "no-store"},
        )

    @router.post("/vision/find")
    async def find_named_object(body: FindRequest):
        return await call("find", **body.model_dump())

    @router.post("/collect")
    async def collect_remaining(body: CollectRequest):
        return await call("collect_remaining", **body.model_dump())

    @router.post("/reset")
    async def reset(body: ResetRequest):
        return await call("reset", **body.model_dump())

    @router.post("/step")
    async def step(body: StepRequest):
        return await call("step", **body.model_dump())

    @router.post("/manual")
    async def manual(body: ManualRequest):
        return await call("jog", **body.model_dump())

    @router.post("/command")
    async def command(body: CommandRequest):
        return await call("command", **body.model_dump())

    @router.post("/record")
    async def record(body: RecordRequest):
        return await call("record", **body.model_dump())

    @router.get("/episodes")
    async def episodes():
        return (await call("snapshot"))["episodes"]

    @router.post("/replay/{identifier}")
    async def load_replay(identifier: UUID):
        if not (directory / f"{identifier}.zip").is_file():
            raise HTTPException(404, "Episode not found")
        return await call("load_replay", identifier=str(identifier))

    @router.post("/replay")
    async def replay(body: ReplayRequest):
        return await call("replay_command", **body.model_dump())

    @router.get("/episodes/{identifier}/download")
    async def download(identifier: UUID):
        path = directory / f"{identifier}.zip"
        if not path.is_file():
            raise HTTPException(404, "Episode not found")
        return FileResponse(path, media_type="application/zip", filename=path.name)

    @router.get("/episodes/{identifier}/frames/{camera}/{index}")
    async def frame(identifier: UUID, camera: Literal["overhead", "wrist"], index: int):
        if not 0 <= index <= 800:
            raise HTTPException(404, "Frame out of range")
        path = directory / f"{identifier}.zip"
        try:
            with zipfile.ZipFile(path) as archive:
                name = f"rgb/{camera}/{index:05d}.png"
                if archive.getinfo(name).file_size > 1_000_000:
                    raise HTTPException(413, "Frame exceeds size limit")
                data = archive.read(name)
        except (OSError, KeyError, zipfile.BadZipFile) as error:
            raise HTTPException(404, "Recorded RGB frame not available") from error
        return Response(data, media_type="image/png", headers={"X-Content-Type-Options": "nosniff"})

    @router.get("/license", response_class=PlainTextResponse)
    async def license_text():
        from .assets import require_model

        try:
            path = await asyncio.to_thread(require_model, cache)
        except FileNotFoundError as error:
            raise HTTPException(503, str(error)) from error
        return (path.parent / "LICENSE").read_text()

    return router
