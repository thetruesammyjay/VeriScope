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
class BraveSearchClient:
    """Adapter for Brave Search's public web-search API."""

    api_key: str
    endpoint: str = "https://api.search.brave.com/res/v1/web/search"
    timeout_seconds: float = 15.0

    def search(
        self,
        query: str,
        *,
        max_results: int = 10,
        recency_days: int | None = None,
    ) -> list[SearchResult]:
        params = {"q": query, "count": max_results}
        if recency_days is not None:
            params["freshness"] = _brave_freshness(recency_days)
        response = httpx.get(
            self.endpoint,
            params=params,
            headers={"Accept": "application/json", "X-Subscription-Token": self.api_key},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        values = response.json().get("web", {}).get("results", [])
        return [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("description", ""),
                published_at=_parse_datetime(item.get("page_age")),
                source_name=item.get("profile", {}).get("long_name"),
            )
            for item in values
            if item.get("url")
        ]


def _brave_freshness(days: int) -> str:
    """Map the pipeline's recency window to Brave's freshness filters."""

    if days <= 1:
        return "pd"
    if days <= 7:
        return "pw"
    if days <= 31:
        return "pm"
    return "py"


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


__all__ = [
    "BraveSearchClient",
    "EmptySearchClient",
    "InMemorySearchClient",
    "SearchClient",
    "SearchResult",
]
