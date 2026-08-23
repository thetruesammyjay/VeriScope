"""Schemas for combined classification and current-source analysis."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .prediction import PredictionResponse
from .verification import VerificationResponse


class AnalysisRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)


class AnalysisResponse(BaseModel):
    prediction: PredictionResponse
    verification: VerificationResponse


__all__ = ["AnalysisRequest", "AnalysisResponse"]
