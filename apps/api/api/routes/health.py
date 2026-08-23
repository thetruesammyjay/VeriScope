"""Service health and runtime information endpoint."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel

from apps.api.core.config import Settings

router = APIRouter(tags=["system"])


class HealthResponse(BaseModel):
    """Public health information safe for a load balancer to consume."""

    status: Literal["ok"]
    environment: str
    model_name: str


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Report that the API process is running and expose its environment."""

    settings: Settings = request.app.state.settings
    return HealthResponse(
        status="ok",
        environment=settings.app_env,
        model_name=settings.model_name,
    )


__all__ = ["HealthResponse", "health", "router"]
