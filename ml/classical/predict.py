"""Inference helpers for persisted classical model artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from .features import normalize_text


@dataclass(frozen=True)
class ClassicalPrediction:
    label: str
    confidence: float
    model: str
    model_version: str
    processing_time_ms: float
    disclaimer: str = (
        "This is a machine-learning prediction and should not be treated as "
        "independent factual verification."
    )


class ClassicalPredictor:
    """Run predictions against a loaded joblib artifact."""

    def __init__(self, artifact: dict[str, Any]) -> None:
        try:
            self._pipeline = artifact["pipeline"]
            metadata = artifact["metadata"]
            self.model = str(metadata["model"])
            self.model_version = str(metadata["model_version"])
        except (KeyError, TypeError) as error:
            raise ValueError("Invalid classical model artifact") from error

    def predict(self, text: str) -> ClassicalPrediction:
        started = perf_counter()
        normalized = normalize_text(text)
        label = str(self._pipeline.predict([normalized])[0])
        probabilities = self._pipeline.predict_proba([normalized])[0]
        classes = [str(value) for value in self._pipeline.classes_]
        confidence = float(probabilities[classes.index(label)])
        return ClassicalPrediction(
            label=label,
            confidence=confidence,
            model=self.model,
            model_version=self.model_version,
            processing_time_ms=(perf_counter() - started) * 1_000,
        )


__all__ = ["ClassicalPrediction", "ClassicalPredictor"]
