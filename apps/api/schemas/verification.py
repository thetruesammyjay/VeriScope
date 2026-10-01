"""HTTP response models for evidence verification."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

EvidenceStatus = Literal["supported", "contradicted", "mixed", "insufficient"]
TransformerReviewStatus = Literal[
    "supported", "contradicted", "mixed", "insufficient", "sources_found"
]


class EvidencePassageResponse(BaseModel):
    url: str
    text: str
    relevance_score: float = Field(ge=0, le=1)
    title: str | None = None
    source_name: str | None = None
    published_at: str | None = None
    retrieved_at: str | None = None
    source_classification: Literal["primary_official", "unclassified"] = "unclassified"


class ClaimAssessmentResponse(BaseModel):
    claim_id: str
    claim: str
    status: TransformerReviewStatus
    evidence: list[EvidencePassageResponse] = Field(default_factory=list)
    rationale: str | None = None


class VerificationResponse(BaseModel):
    status: TransformerReviewStatus
    claims: list[ClaimAssessmentResponse] = Field(default_factory=list)
    review_mode: Literal["standard", "transformer_retrieval"] = "standard"


__all__ = [
    "ClaimAssessmentResponse",
    "EvidencePassageResponse",
    "EvidenceStatus",
    "TransformerReviewStatus",
    "VerificationResponse",
]
