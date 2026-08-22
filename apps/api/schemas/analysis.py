"""Schemas for the classification-plus-current-evidence workflow."""

from __future__ import annotations

from dataclasses import dataclass, field

from .claim import ClaimAssessmentSchema
from .evidence import EvidenceSchema


@dataclass(frozen=True)
class AnalysisRequest:
    text: str


@dataclass(frozen=True)
class AnalysisResponse:
    label: str
    confidence: float
    model: str
    model_version: str
    processing_time_ms: float
    evidence: EvidenceSchema
    claims: tuple[ClaimAssessmentSchema, ...] = field(default_factory=tuple)
    disclaimer: str = (
        "This is a machine-learning and evidence-retrieval assessment, not independent factual verification."
    )

