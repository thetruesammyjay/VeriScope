"""Evidence-aware article analysis routes."""

from __future__ import annotations

import logging
from typing import Annotated

from apps.api.api.dependencies import (
    get_deepseek_explainer,
    get_inference_service,
    get_verification_pipeline,
)
from apps.api.core.config import Settings
from apps.api.schemas.analysis import (
    AnalysisRequest,
    AnalysisResponse,
    ExplanationResponse,
    validate_article_text,
)
from apps.api.schemas.prediction import PredictionResponse
from apps.api.schemas.verification import (
    ClaimAssessmentResponse,
    EvidencePassageResponse,
    VerificationResponse,
)
from apps.api.services.deepseek_explainer import (
    DeepSeekExplainer,
    DeepSeekExplanationError,
)
from apps.api.services.inference_service import InferenceService
from fastapi import APIRouter, Depends, HTTPException, Request
from ml.verification.pipeline import VerificationPipeline

router = APIRouter(prefix="/api/v1", tags=["analysis"])
logger = logging.getLogger(__name__)


@router.post("/analyze", response_model=AnalysisResponse)
def analyze(
    request: AnalysisRequest,
    http_request: Request,
    pipeline: Annotated[
        VerificationPipeline, Depends(get_verification_pipeline)
    ],
    inference: Annotated[InferenceService, Depends(get_inference_service)],
    explainer: Annotated[
        DeepSeekExplainer | None, Depends(get_deepseek_explainer)
    ],
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
    explanation_response = ExplanationResponse(
        available=False,
        error=(
            "DeepSeek explanation is not configured."
            if explainer is None
            else "A prediction is required to generate an explanation."
        ),
    )
    if explainer is not None and prediction is not None:
        try:
            explanation = explainer.explain(
                label=prediction.label,
                confidence=prediction.confidence,
                confidence_method=getattr(prediction, "confidence_method", None),
                verification_status=summary.status,
                article_text=article_text,
            )
            explanation_response = ExplanationResponse(
                available=True,
                text=explanation.text,
                model=explanation.model,
            )
        except DeepSeekExplanationError:
            # A provider failure must not discard the local prediction or
            # retrieved evidence, and should not expose provider response data.
            logger.warning("DeepSeek explanation request failed")
            explanation_response = ExplanationResponse(
                available=False,
                error="DeepSeek explanation is unavailable right now.",
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
        explanation=explanation_response,
    )


__all__ = ["analyze", "router"]
