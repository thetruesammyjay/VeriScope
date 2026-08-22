"""Extract checkable claims from article text."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Claim:
    """A factual statement selected for external verification."""

    claim_id: str
    text: str
    article_span: tuple[int, int] | None = None


def extract_claims(text: str, *, max_claims: int = 5) -> list[Claim]:
    """Return claim candidates.

    This scaffold intentionally performs no semantic extraction yet. The
    production implementation will use a documented model or rule-based
    strategy and will preserve the source span for traceability.
    """

    del text, max_claims
    return []

