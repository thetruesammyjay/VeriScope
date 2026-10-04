"""FastAPI dependency providers for retrieval and verification services."""

from __future__ import annotations

from functools import lru_cache

from apps.api.core.config import Settings, get_settings
from apps.api.services.claim_question import (
    ClaimQuestionService,
    DeepSeekClaimAnswerer,
)
from apps.api.services.deepseek_explainer import DeepSeekExplainer
from apps.api.services.inference_service import InferenceService
from fastapi import Request
from ml.classical.predict import ClassicalPredictor
from ml.inference.loader import load_artifact
from ml.retrieval.document_fetcher import HttpDocumentFetcher
from ml.retrieval.search_client import BraveSearchClient, EmptySearchClient, SearchClient
from ml.retrieval.source_filter import SourcePolicy
from ml.verification.pipeline import VerificationPipeline


def build_search_client(settings: Settings) -> SearchClient:
    """Select a live provider only when its required configuration exists."""

    if (
        settings.search_provider
        and settings.search_provider.lower() == "brave"
        and settings.search_api_key
    ):
        return BraveSearchClient(
            endpoint=settings.search_endpoint or BraveSearchClient.endpoint,
            api_key=settings.search_api_key,
            timeout_seconds=settings.search_timeout_seconds,
        )
    return EmptySearchClient()


@lru_cache(maxsize=1)
def get_verification_pipeline() -> VerificationPipeline:
    settings = get_settings()
    return VerificationPipeline(
        search_client=build_search_client(settings),
        source_policy=SourcePolicy(blocked_domains=("wikipedia.org",)),
        document_fetcher=HttpDocumentFetcher(
            timeout_seconds=settings.search_timeout_seconds,
        ),
        max_search_results=settings.search_max_results,
        max_claims=settings.evidence_max_claims,
        max_sources=settings.evidence_max_sources,
        recency_days=settings.evidence_recency_days,
    )


def load_transformer_predictor(path):
    """Import PyTorch/Transformers only for a transformer deployment."""

    from ml.transformer.predict import TransformerPredictor

    return TransformerPredictor(path)


def build_inference_service(settings: Settings) -> InferenceService:
    """Load the predictor selected by ``PRODUCTION_MODEL`` when available."""

    try:
        if settings.production_model == "transformer":
            return InferenceService(load_transformer_predictor(settings.transformer_model_path))
        return InferenceService(ClassicalPredictor(load_artifact(settings.classical_model_path)))
    except (FileNotFoundError, ImportError, OSError, ValueError):
        return InferenceService()


def build_deepseek_explainer(settings: Settings) -> DeepSeekExplainer | None:
    """Enable optional explanations only when a non-empty secret is set."""

    if settings.deepseek_api_key is None:
        return None
    api_key = settings.deepseek_api_key.get_secret_value().strip()
    if not api_key:
        return None
    return DeepSeekExplainer(
        api_key=api_key,
        model=settings.deepseek_model,
        timeout_seconds=settings.deepseek_timeout_seconds,
        max_input_chars=settings.deepseek_max_input_chars,
        max_output_tokens=settings.deepseek_max_output_tokens,
    )


def build_deepseek_claim_answerer(
    settings: Settings,
) -> DeepSeekClaimAnswerer | None:
    """Enable claim answers only when a non-empty server-side key is set."""

    if settings.deepseek_api_key is None:
        return None
    api_key = settings.deepseek_api_key.get_secret_value().strip()
    if not api_key:
        return None
    return DeepSeekClaimAnswerer(
        api_key=api_key,
        model=settings.deepseek_model,
        timeout_seconds=settings.deepseek_timeout_seconds,
        max_input_chars=settings.deepseek_max_input_chars,
        max_output_tokens=settings.deepseek_question_max_output_tokens,
    )


def get_claim_question_service(request: Request) -> ClaimQuestionService:
    """Build the isolated claim-Q&A workflow from this app's settings."""

    settings: Settings = request.app.state.settings
    return ClaimQuestionService(
        search_client=build_search_client(settings),
        document_fetcher=HttpDocumentFetcher(
            timeout_seconds=min(settings.search_timeout_seconds, 6.0),
        ),
        answerer=build_deepseek_claim_answerer(settings),
        source_policy=SourcePolicy(blocked_domains=("wikipedia.org",)),
        max_sources=4,
        max_input_chars=settings.deepseek_max_input_chars,
    )


@lru_cache(maxsize=1)
def get_inference_service() -> InferenceService:
    """Return the configured production inference service for this process."""

    return build_inference_service(get_settings())


def get_deepseek_explainer() -> DeepSeekExplainer | None:
    """Resolve optional DeepSeek explanation service from environment config."""

    return build_deepseek_explainer(get_settings())


__all__ = [
    "build_search_client",
    "build_inference_service",
    "build_deepseek_explainer",
    "build_deepseek_claim_answerer",
    "get_claim_question_service",
    "get_deepseek_explainer",
    "get_inference_service",
    "get_verification_pipeline",
    "load_transformer_predictor",
]
