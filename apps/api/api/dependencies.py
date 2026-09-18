"""FastAPI dependency providers for retrieval and verification services."""

from __future__ import annotations

from functools import lru_cache

from apps.api.core.config import Settings, get_settings
from apps.api.services.inference_service import InferenceService
from ml.classical.predict import ClassicalPredictor
from ml.inference.loader import load_artifact
from ml.retrieval.document_fetcher import HttpDocumentFetcher
from ml.retrieval.search_client import BingSearchClient, EmptySearchClient, SearchClient
from ml.verification.pipeline import VerificationPipeline


def build_search_client(settings: Settings) -> SearchClient:
    """Select a live provider only when its required configuration exists."""

    if (
        settings.search_provider
        and settings.search_provider.lower() == "bing"
        and settings.search_endpoint
        and settings.search_api_key
    ):
        return BingSearchClient(
            endpoint=settings.search_endpoint,
            api_key=settings.search_api_key,
            timeout_seconds=settings.search_timeout_seconds,
        )
    return EmptySearchClient()


@lru_cache(maxsize=1)
def get_verification_pipeline() -> VerificationPipeline:
    settings = get_settings()
    return VerificationPipeline(
        search_client=build_search_client(settings),
        document_fetcher=HttpDocumentFetcher(
            timeout_seconds=settings.search_timeout_seconds,
        ),
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
        return InferenceService(
            ClassicalPredictor(load_artifact(settings.classical_model_path))
        )
    except (FileNotFoundError, ImportError, OSError, ValueError):
        return InferenceService()


@lru_cache(maxsize=1)
def get_inference_service() -> InferenceService:
    """Return the configured production inference service for this process."""

    return build_inference_service(get_settings())


__all__ = [
    "build_search_client",
    "build_inference_service",
    "get_inference_service",
    "get_verification_pipeline",
    "load_transformer_predictor",
]
