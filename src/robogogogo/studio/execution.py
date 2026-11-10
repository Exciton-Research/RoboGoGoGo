"""Local operator API for the explicitly enabled command test backend."""

import asyncio
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from robogogogo.robot.execution import CoordinatedGoal, GripperGoal

router = APIRouter(prefix="/api/execution", tags=["command execution test"])


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)


class ArmRequest(StrictRequest):
    owner: Literal["manual", "policy"]


class LeaseRequest(StrictRequest):
    token: SecretStr


class GripperRequest(StrictRequest):
    width: float
    force_limit: float


class CommandRequest(LeaseRequest):
    sequence: int = Field(ge=1)
    observation_sequence: int = Field(ge=1)
    left: list[float] = Field(min_length=6, max_length=6)
    right: list[float] = Field(min_length=6, max_length=6)
    left_gripper: GripperRequest
    right_gripper: GripperRequest
    duration_s: float = Field(ge=0.1, le=10)


def service(request):
    executor = request.app.state.execution
    if executor is None:
        raise HTTPException(
            503, "Command test backend is disabled; start Studio with --test-execution"
        )
    return executor


@router.get("")
async def status(request: Request):
    if request.app.state.execution is None:
        return dict(
            backend=None,
            mode="disabled",
            physical_execution_enabled=False,
            reason="Start Studio with --test-execution to exercise command control without a robot",
        )
    return await asyncio.to_thread(service(request).snapshot)


@router.post("/arm")
async def arm(body: ArmRequest, request: Request):
    return await asyncio.to_thread(service(request).arm, body.owner)


@router.post("/heartbeat")
async def heartbeat(body: LeaseRequest, request: Request):
    return await asyncio.to_thread(service(request).heartbeat, body.token.get_secret_value())


@router.post("/command")
async def command(body: CommandRequest, request: Request):
    goal = CoordinatedGoal(
        tuple(body.left),
        tuple(body.right),
        GripperGoal(**body.left_gripper.model_dump()),
        GripperGoal(**body.right_gripper.model_dump()),
        body.duration_s,
    )
    return await asyncio.to_thread(
        service(request).submit,
        body.token.get_secret_value(),
        body.sequence,
        body.observation_sequence,
        goal,
    )


@router.post("/stop")
async def stop(request: Request):
    return await asyncio.to_thread(service(request).stop)


@router.post("/takeover")
async def takeover(request: Request):
    return await asyncio.to_thread(service(request).takeover)


@router.post("/reset")
async def reset(request: Request):
    return await asyncio.to_thread(service(request).reset)
