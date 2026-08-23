"""HTTP response models for evidence verification."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

EvidenceStatus = Literal["supported", "contradicted", "mixed", "insufficient"]


class EvidencePassageResponse(BaseModel):
    url: str
    text: str
    relevance_score: float = Field(ge=0, le=1)
    title: str | None = None
    source_name: str | None = None
    published_at: str | None = None
    retrieved_at: str | None = None


class ClaimAssessmentResponse(BaseModel):
    claim_id: str
    claim: str
    status: EvidenceStatus
    evidence: list[EvidencePassageResponse] = Field(default_factory=list)
    rationale: str | None = None


class VerificationResponse(BaseModel):
    status: EvidenceStatus
    claims: list[ClaimAssessmentResponse] = Field(default_factory=list)


__all__ = [
    "ClaimAssessmentResponse",
    "EvidencePassageResponse",
    "EvidenceStatus",
    "VerificationResponse",
]
