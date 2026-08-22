"""Application adapter for production model inference."""

from __future__ import annotations


class InferenceService:
    """Placeholder contract consumed by the analysis service."""

    def predict(self, article_text: str):
        """Return a prediction once the ML inference engine is implemented."""

        del article_text
        raise NotImplementedError("Production inference is not implemented yet.")
