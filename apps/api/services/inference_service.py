"""Application adapter for production model inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class ProductionPredictor(Protocol):
    """The shared prediction surface for classical and transformer artifacts."""

    def predict(self, text: str) -> Any:
        """Return a prediction with the public API fields."""


@dataclass
class InferenceService:
    """Run the selected production predictor when its artifact is available."""

    predictor: ProductionPredictor | None = None

    @property
    def available(self) -> bool:
        return self.predictor is not None

    def predict(self, article_text: str) -> Any | None:
        """Return a prediction, or ``None`` until an artifact is trained."""

        if self.predictor is None:
            return None
        return self.predictor.predict(article_text)


__all__ = ["InferenceService", "ProductionPredictor"]
