"""Contracts for fetching and normalising retrieved source pages."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from typing import ClassVar, Protocol

import httpx


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


class _VisibleTextParser(HTMLParser):
    """Small dependency-free HTML-to-text parser for retrieved pages."""

    _ignored_tags: ClassVar[set[str]] = {
        "script",
        "style",
        "noscript",
        "svg",
        "template",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag.lower() in self._ignored_tags:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._ignored_tags and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            cleaned = " ".join(data.split())
            if cleaned:
                self.parts.append(cleaned)


def _html_to_text(html: str) -> str:
    parser = _VisibleTextParser()
    parser.feed(html)
    return " ".join(parser.parts)


@dataclass
class HttpDocumentFetcher:
    """Fetch public HTML pages with bounded time and response size."""

    timeout_seconds: float = 15.0
    max_bytes: int = 2_000_000
    user_agent: str = "Automated-Fake-News-Detection/0.1"

    def fetch(self, url: str) -> RetrievedDocument:
        with httpx.Client(
            timeout=self.timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": self.user_agent},
        ) as client:
            response = client.get(url)
            response.raise_for_status()
            content = response.content[: self.max_bytes]

        encoding = response.encoding or "utf-8"
        content_type = response.headers.get("content-type", "")
        decoded = content.decode(encoding, errors="replace")
        text = _html_to_text(decoded) if "html" in content_type.lower() else decoded
        return RetrievedDocument(
            url=str(response.url),
            title=response.url.host or url,
            text=text,
            retrieved_at=datetime.now(UTC).isoformat(),
        )


@dataclass
class InMemoryDocumentFetcher:
    """Deterministic document fetcher for tests and local demonstrations."""

    documents: dict[str, RetrievedDocument]

    def fetch(self, url: str) -> RetrievedDocument:
        return self.documents[url]


__all__ = [
    "DocumentFetcher",
    "HttpDocumentFetcher",
    "InMemoryDocumentFetcher",
    "RetrievedDocument",
]
