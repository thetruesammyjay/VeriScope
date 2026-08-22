"""Coordinate model inference and current-source verification."""

from __future__ import annotations

from dataclasses import dataclass

from .inference_service import InferenceService
from .verification_service import VerificationService


@dataclass
class AnalysisService:
    inference_service: InferenceService
    verification_service: VerificationService

    def analyze(self, article_text: str):
        """Return a combined analysis once both service implementations exist."""

        prediction = self.inference_service.predict(article_text)
        verification = self.verification_service.verify(article_text)
        return prediction, verification

