"""HTTP response model for classical model predictions."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    available: bool
    label: Literal["likely_real", "likely_fake"] | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    model: str | None = None
    model_version: str | None = None
    processing_time_ms: float | None = Field(default=None, ge=0)
    error: str | None = None
    disclaimer: str = (
        "This is a machine-learning prediction and should not be treated as "
        "independent factual verification."
    )


__all__ = ["PredictionResponse"]
