"""Application adapter for current-source retrieval."""

from __future__ import annotations

from dataclasses import dataclass

from ml.retrieval.document_fetcher import DocumentFetcher
from ml.retrieval.search_client import SearchClient, SearchResult
from ml.retrieval.source_filter import SourcePolicy, filter_sources


@dataclass
class RetrievalService:
    search_client: SearchClient
    document_fetcher: DocumentFetcher | None = None
    source_policy: SourcePolicy | None = None

    def search(self, query: str, *, max_results: int = 10) -> list[SearchResult]:
        results = self.search_client.search(query, max_results=max_results)
        return filter_sources(results, self.source_policy)

