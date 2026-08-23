"""Evidence-aware article analysis routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from apps.api.api.dependencies import get_verification_pipeline
from apps.api.schemas.verification import (
    ClaimAssessmentResponse,
    EvidencePassageResponse,
    VerificationResponse,
)
from ml.verification.pipeline import VerificationPipeline

router = APIRouter(prefix="/api/v1", tags=["analysis"])


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)


@router.post("/analyze", response_model=VerificationResponse)
def analyze(
    request: AnalyzeRequest,
    pipeline: Annotated[
        VerificationPipeline, Depends(get_verification_pipeline)
    ],
) -> VerificationResponse:
    """Return current-source evidence assessments for an article."""

    summary = pipeline.verify(request.text)
    claims = [
        ClaimAssessmentResponse(
            claim_id=assessment.claim.claim_id,
            claim=assessment.claim.text,
            status=assessment.status,
            rationale=assessment.rationale,
            evidence=[
                EvidencePassageResponse(
                    url=passage.document_url,
                    text=passage.text,
                    relevance_score=passage.relevance_score,
                    title=passage.title,
                    source_name=passage.source_name,
                    published_at=passage.published_at,
                    retrieved_at=passage.retrieved_at,
                )
                for passage in assessment.evidence
            ],
        )
        for assessment in summary.assessments
    ]
    return VerificationResponse(status=summary.status, claims=claims)


__all__ = ["AnalyzeRequest", "analyze", "router"]
