"""Metrics for claim-verdict and end-to-end verification experiments."""

from __future__ import annotations


def verification_metrics(*args: object, **kwargs: object) -> dict[str, float | None]:
    """Return the future verification-evaluation schema."""

    del args, kwargs
    return {
        "verdict_accuracy": None,
        "macro_f1": None,
        "evidence_supported_accuracy": None,
        "insufficient_evidence_rate": None,
    }

