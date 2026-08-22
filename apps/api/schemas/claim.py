"""Claim and claim-assessment response schemas."""

from __future__ import annotations

from dataclasses import dataclass, field

from .source import SourceSchema


@dataclass(frozen=True)
class ClaimSchema:
    claim_id: str
    text: str


@dataclass(frozen=True)
class ClaimAssessmentSchema:
    claim: ClaimSchema
    status: str
    evidence_text: str | None = None
    sources: tuple[SourceSchema, ...] = field(default_factory=tuple)
    rationale: str | None = None

