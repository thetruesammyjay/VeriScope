"""Orchestrate claim extraction, retrieval, and evidence verification."""

from __future__ import annotations

from dataclasses import dataclass

from ml.retrieval.search_client import SearchClient

from .aggregator import VerificationSummary
from .claim_extractor import Claim, extract_claims


@dataclass
class VerificationPipeline:
    """Dependency-injection boundary for the evidence workflow."""

    search_client: SearchClient
    max_claims: int = 5

    def extract_claims(self, article_text: str) -> list[Claim]:
        return extract_claims(article_text, max_claims=self.max_claims)

    def verify(self, article_text: str) -> VerificationSummary:
        """Run the workflow once dependencies are implemented."""

        del article_text
        raise NotImplementedError(
            "Claim search, document fetching, evidence extraction, and verification are not implemented yet."
        )

