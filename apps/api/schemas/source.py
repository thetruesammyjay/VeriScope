"""Source metadata returned by evidence retrieval."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceSchema:
    title: str
    url: str
    source_name: str | None = None
    published_at: str | None = None
    retrieved_at: str | None = None

