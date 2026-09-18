"""Evidence-aware article analysis routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request

from apps.api.api.dependencies import get_inference_service, get_verification_pipeline
from apps.api.core.config import Settings
from apps.api.schemas.analysis import (
    AnalysisRequest,
    AnalysisResponse,
    validate_article_text,
)
from apps.api.schemas.prediction import PredictionResponse
from apps.api.schemas.verification import (
    ClaimAssessmentResponse,
    EvidencePassageResponse,
    VerificationResponse,
)
from apps.api.services.inference_service import InferenceService
from ml.verification.pipeline import VerificationPipeline

router = APIRouter(prefix="/api/v1", tags=["analysis"])


@router.post("/analyze", response_model=AnalysisResponse)
def analyze(
    request: AnalysisRequest,
    http_request: Request,
    pipeline: Annotated[
        VerificationPipeline, Depends(get_verification_pipeline)
    ],
    inference: Annotated[InferenceService, Depends(get_inference_service)],
) -> AnalysisResponse:
    """Return the classical prediction and current-source assessment."""

    settings: Settings = http_request.app.state.settings
    try:
        article_text = validate_article_text(
            request.text,
            min_length=settings.min_article_length,
            max_length=settings.max_article_length,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    summary = pipeline.verify(article_text)
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
    prediction = inference.predict(article_text)
    if prediction is None:
        prediction_response = PredictionResponse(
            available=False,
            error="Classical model artifact is not available. Train and deploy it first.",
        )
    else:
        prediction_response = PredictionResponse(
            available=True,
            label=prediction.label,
            confidence=prediction.confidence,
            model=prediction.model,
            model_version=prediction.model_version,
            processing_time_ms=prediction.processing_time_ms,
            disclaimer=prediction.disclaimer,
        )
    return AnalysisResponse(
        prediction=prediction_response,
        verification=VerificationResponse(status=summary.status, claims=claims),
    )


__all__ = ["analyze", "router"]
