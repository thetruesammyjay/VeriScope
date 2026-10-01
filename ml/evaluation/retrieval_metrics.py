"""Metrics for query and evidence retrieval experiments."""

from __future__ import annotations

from collections.abc import Sequence
from urllib.parse import urlsplit

from ml.retrieval.source_filter import canonical_source_url


def retrieval_metrics(
    retrieved_urls: Sequence[str],
    relevant_urls: Sequence[str],
    *,
    k: int = 5,
) -> dict[str, float | None]:
    """Calculate URL-level precision, recall, coverage, and source diversity.

    Duplicate URLs are collapsed before scoring. Precision divides by the
    number actually returned up to ``k`` (rather than padding missing results
    to ``k``). ``relevant_urls`` is the hand-labelled set of pages that
    genuinely address the claim.
    """

    if k < 1:
        raise ValueError("k must be at least 1")
    retrieved = _unique_urls(retrieved_urls)[:k]
    relevant = set(_unique_urls(relevant_urls))
    if not relevant:
        raise ValueError("each evaluation claim must have at least one relevant URL")
    hits = len(set(retrieved) & relevant)
    domains = {
        (urlsplit(url).hostname or "").lower().removeprefix("www.")
        for url in retrieved
    }
    return {
        "claim_coverage": float(hits > 0),
        "evidence_precision_at_k": hits / min(k, len(retrieved)) if retrieved else 0.0,
        "evidence_recall_at_k": hits / len(relevant),
        "source_diversity": len(domains) / len(retrieved) if retrieved else 0.0,
        "relevant_sources_retrieved": float(hits),
    }


def _unique_urls(urls: Sequence[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for url in urls:
        parsed = urlsplit(url)
        key = canonical_source_url(url)
        if parsed.hostname and key not in seen:
            seen.add(key)
            unique.append(key)
    return unique


__all__ = ["retrieval_metrics"]
