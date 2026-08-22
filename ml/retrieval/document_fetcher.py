"""Contracts for fetching and normalising retrieved source pages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RetrievedDocument:
    """Normalised source content used by evidence extraction."""

    url: str
    title: str
    text: str
    published_at: str | None = None
    retrieved_at: str | None = None
    source_name: str | None = None


class DocumentFetcher(Protocol):
    """Interface for an HTTP/page-content adapter."""

    def fetch(self, url: str) -> RetrievedDocument:
        """Fetch and normalise one public source document."""

