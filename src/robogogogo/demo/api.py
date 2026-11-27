"""Local demo API; initialize MuJoCo lazily without downloading assets."""

import asyncio
import threading
from contextlib import asynccontextmanager

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field


class DemoCommand(BaseModel):
    command: str
    variant: int = Field(default=0, ge=0, le=1000000)


def create_demo_router(cache, *, legacy_workspace=False):
    task = None
    lock = threading.Lock()

    def get_task():
        nonlocal task
        with lock:
            if task is None:
                try:
                    from .task import PickPlace

                    task = PickPlace(cache)
                except (ImportError, FileNotFoundError) as error:
                    raise HTTPException(
                        503,
                        "Demo setup needed: install .[studio,demo] and run robogogogo-demo --setup. "
                        + str(error),
                    ) from error
            return task

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            if task is not None:
                await asyncio.to_thread(task.close)

    router = APIRouter(prefix="/api/demo", lifespan=lifespan)

    @router.get("/scene")
    async def scene():
        return {
            **await asyncio.to_thread(lambda: get_task().geometry()),
            "legacy_workspace": legacy_workspace,
        }

    @router.get("/state")
    async def state():
        return await asyncio.to_thread(lambda: get_task().snapshot())

    @router.get("/license", response_class=PlainTextResponse)
    async def license_text():
        from .assets import require_model

        try:
            path = await asyncio.to_thread(require_model, cache)
        except FileNotFoundError as error:
            raise HTTPException(503, str(error)) from error
        return (path.parent / "LICENSE").read_text()

    @router.post("/command")
    async def command(body: DemoCommand):
        try:
            return await asyncio.to_thread(lambda: get_task().command(body.command, body.variant))
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    return router
