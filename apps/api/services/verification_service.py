"""Application adapter for claim and evidence verification."""

from __future__ import annotations

from dataclasses import dataclass

from ml.verification.pipeline import VerificationPipeline


@dataclass
class VerificationService:
    pipeline: VerificationPipeline

    def verify(self, article_text: str):
        return self.pipeline.verify(article_text)

