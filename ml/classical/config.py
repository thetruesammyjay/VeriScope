"""Configuration for the TF-IDF and Logistic Regression baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ClassicalConfig:
    max_features: int = 50_000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 1
    max_df: float = 0.95
    max_iter: int = 1_000
    random_state: int = 42
    model_version: str = "classical-tfidf-logreg-0.1.0"
    artifact_path: Path = Path("models/classical/model.joblib")


__all__ = ["ClassicalConfig"]
