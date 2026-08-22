"""Metrics for query and evidence retrieval experiments."""

from __future__ import annotations


def retrieval_metrics(*args: object, **kwargs: object) -> dict[str, float | None]:
    """Return the future retrieval-evaluation schema."""

    del args, kwargs
    return {
        "claim_coverage": None,
        "evidence_precision_at_k": None,
        "evidence_recall_at_k": None,
        "source_diversity": None,
    }

