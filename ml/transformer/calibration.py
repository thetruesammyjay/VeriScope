"""Temperature scaling utilities for held-out logits."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp


def negative_log_likelihood(logits: np.ndarray, labels: np.ndarray) -> float:
    shifted = logits - logsumexp(logits, axis=1, keepdims=True)
    return float(-shifted[np.arange(len(labels)), labels].mean())


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit one positive scalar by minimizing calibration-set NLL."""

    values = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or len(values) != len(targets) or not len(targets):
        raise ValueError("logits must be a non-empty 2D array aligned with labels")
    if np.any(~np.isfinite(values)) or np.any((targets < 0) | (targets >= values.shape[1])):
        raise ValueError("logits or labels contain invalid values")

    result = minimize_scalar(
        lambda log_temperature: negative_log_likelihood(values / np.exp(log_temperature), targets),
        bounds=(-4.0, 4.0),
        method="bounded",
    )
    if not result.success:
        raise RuntimeError("temperature fitting did not converge")
    return float(np.exp(result.x))


__all__ = ["fit_temperature", "negative_log_likelihood"]
