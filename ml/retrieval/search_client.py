"""Provider-neutral search client contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import httpx


@dataclass(frozen=True)
class SearchResult:
    """A result returned by a search provider."""

    title: str
    url: str
    snippet: str = ""
    published_at: datetime | None = None
    source_name: str | None = None


class SearchClient(Protocol):
    """Interface implemented by a concrete search-provider adapter."""

    def search(
        self,
        query: str,
        *,
        max_results: int = 10,
        recency_days: int | None = None,
    ) -> list[SearchResult]:
        """Return candidate sources for a query."""


@dataclass
class EmptySearchClient:
    """Safe default used when no live search provider is configured."""

    def search(
        self,
        query: str,
        *,
        max_results: int = 10,
        recency_days: int | None = None,
    ) -> list[SearchResult]:
        del query, max_results, recency_days
        return []


@dataclass
class InMemorySearchClient:
    """Deterministic search client for unit and integration tests."""

    results: list[SearchResult]

    def search(
        self,
        query: str,
        *,
        max_results: int = 10,
        recency_days: int | None = None,
    ) -> list[SearchResult]:
        del recency_days
        terms = {term.lower() for term in query.split() if len(term) > 2}
        ranked = sorted(
            self.results,
            key=lambda result: len(terms & set(result.title.lower().split())),
            reverse=True,
        )
        return ranked[:max_results]


@dataclass
class BingSearchClient:
    """Adapter for Bing-compatible web search JSON responses."""

    endpoint: str
    api_key: str
    timeout_seconds: float = 15.0

    def search(
        self,
        query: str,
        *,
        max_results: int = 10,
        recency_days: int | None = None,
    ) -> list[SearchResult]:
        params = {
            "q": query,
            "count": max_results,
            "textDecorations": "false",
            "textFormat": "Raw",
        }
        if recency_days is not None:
            params["freshness"] = _freshness_label(recency_days)
        response = httpx.get(
            self.endpoint,
            params=params,
            headers={"Ocp-Apim-Subscription-Key": self.api_key},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        values = response.json().get("webPages", {}).get("value", [])
        return [
            SearchResult(
                title=item.get("name", ""),
                url=item.get("url", ""),
                snippet=item.get("snippet", ""),
                published_at=_parse_datetime(item.get("dateLastCrawled")),
                source_name=(
                    item.get("provider", [{}])[0].get("name")
                    if item.get("provider")
                    else None
                ),
            )
            for item in values
            if item.get("url")
        ]


def _freshness_label(days: int) -> str:
    if days <= 1:
        return "Day"
    if days <= 7:
        return "Week"
    return "Month"


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


__all__ = [
    "BingSearchClient",
    "EmptySearchClient",
    "InMemorySearchClient",
    "SearchClient",
    "SearchResult",
]
