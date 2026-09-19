"""Contracts for fetching and normalising retrieved source pages."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
import ipaddress
import socket
from typing import ClassVar, Protocol
from urllib.parse import urljoin, urlparse

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


def validate_public_url(url: str) -> str:
    """Reject URLs that could make evidence retrieval reach private services.

    Search results are external input.  Resolving the host before every
    request (including redirects) prevents requests to loopback, private,
    link-local, and other non-public addresses.
    """

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Source URL must use HTTP or HTTPS")
    if not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Source URL must contain a valid public hostname")

    try:
        addresses = socket.getaddrinfo(
            parsed.hostname, None, type=socket.SOCK_STREAM
        )
    except socket.gaierror as error:
        raise ValueError("Source hostname could not be resolved") from error

    if not addresses:
        raise ValueError("Source hostname could not be resolved")
    for address in addresses:
        if not ipaddress.ip_address(address[4][0]).is_global:
            raise ValueError("Source URL must resolve only to public addresses")
    return url


@dataclass
class HttpDocumentFetcher:
    """Fetch public HTML pages with bounded time and response size."""

    timeout_seconds: float = 15.0
    max_bytes: int = 2_000_000
    user_agent: str = "Automated-Fake-News-Detection/0.1"
    max_redirects: int = 5

    def fetch(self, url: str) -> RetrievedDocument:
        with httpx.Client(
            timeout=self.timeout_seconds,
            follow_redirects=False,
            headers={"User-Agent": self.user_agent},
        ) as client:
            current_url = validate_public_url(url)
            for redirect_count in range(self.max_redirects + 1):
                with client.stream("GET", current_url) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            raise ValueError("Source redirect did not provide a location")
                        if redirect_count == self.max_redirects:
                            raise ValueError("Source exceeded the maximum number of redirects")
                        current_url = validate_public_url(urljoin(current_url, location))
                        continue

                    response.raise_for_status()
                    chunks: list[bytes] = []
                    remaining = self.max_bytes
                    for chunk in response.iter_bytes(chunk_size=64 * 1024):
                        chunks.append(chunk[:remaining])
                        remaining -= len(chunk)
                        if remaining <= 0:
                            break
                    content = b"".join(chunks)
                    final_url = str(response.url)
                    title = response.url.host or current_url
                    encoding = response.encoding or "utf-8"
                    content_type = response.headers.get("content-type", "")
                    break
            else:  # pragma: no cover - the range always ends through a branch above
                raise ValueError("Unable to retrieve source")

        decoded = content.decode(encoding, errors="replace")
        text = _html_to_text(decoded) if "html" in content_type.lower() else decoded
        return RetrievedDocument(
            url=final_url,
            title=title,
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
    "validate_public_url",
]
