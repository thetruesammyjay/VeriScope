"""FastAPI dependency providers for retrieval and verification services."""

from __future__ import annotations

from functools import lru_cache

from apps.api.core.config import Settings, get_settings
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


__all__ = ["build_search_client", "get_verification_pipeline"]
