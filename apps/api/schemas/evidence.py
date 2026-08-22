"""Evidence status and source-passage schemas."""

from __future__ import annotations

from dataclasses import dataclass, field

from .source import SourceSchema


@dataclass(frozen=True)
class EvidenceSchema:
    status: str
    summary: str | None = None
    sources: tuple[SourceSchema, ...] = field(default_factory=tuple)

