"""Provider-neutral search client contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


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

