"""Evidence-aware article analysis routes."""

from __future__ import annotations

from typing import Annotated

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
from fastapi import APIRouter, Depends, HTTPException, Request
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
    """Return a model prediction and the matching source-review workflow."""

    settings: Settings = http_request.app.state.settings
    try:
        article_text = validate_article_text(
            request.text,
            min_length=settings.min_article_length,
            max_length=settings.max_article_length,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    prediction = inference.predict(article_text)
    transformer_retrieval = bool(
        prediction and "transformer" in str(prediction.model).lower()
    )
    summary = pipeline.verify(
        article_text,
        transformer_retrieval=transformer_retrieval,
    )
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
                    source_classification=passage.source_classification,
                )
                for passage in assessment.evidence
            ],
        )
        for assessment in summary.assessments
    ]
    if prediction is None:
        prediction_response = PredictionResponse(
            available=False,
            error=(
                "The selected model artifact is not available. Train or download it first."
            ),
        )
    else:
        prediction_response = PredictionResponse(
            available=True,
            label=prediction.label,
            confidence=prediction.confidence,
            confidence_method=getattr(prediction, "confidence_method", None),
            model=prediction.model,
            model_version=prediction.model_version,
            processing_time_ms=prediction.processing_time_ms,
            disclaimer=prediction.disclaimer,
        )
    return AnalysisResponse(
        prediction=prediction_response,
        verification=VerificationResponse(
            status=summary.status,
            claims=claims,
            review_mode=(
                "transformer_retrieval" if transformer_retrieval else "standard"
            ),
        ),
    )


__all__ = ["analyze", "router"]
