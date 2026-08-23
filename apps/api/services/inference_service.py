"""Application adapter for production model inference."""

from __future__ import annotations

from dataclasses import dataclass

from ml.classical.predict import ClassicalPrediction, ClassicalPredictor


@dataclass
class InferenceService:
    """Run the selected classical baseline when an artifact is available."""

    predictor: ClassicalPredictor | None = None

    @property
    def available(self) -> bool:
        return self.predictor is not None

    def predict(self, article_text: str) -> ClassicalPrediction | None:
        """Return a prediction, or ``None`` until an artifact is trained."""

        if self.predictor is None:
            return None
        return self.predictor.predict(article_text)


__all__ = ["InferenceService"]
