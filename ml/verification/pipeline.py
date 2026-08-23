"""Orchestrate claim extraction, retrieval, and evidence verification."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from ml.retrieval.document_fetcher import DocumentFetcher, RetrievedDocument
from ml.retrieval.query_builder import build_queries
from ml.retrieval.ranker import rank_documents
from ml.retrieval.search_client import SearchClient
from ml.retrieval.source_filter import SourcePolicy, filter_sources

from .aggregator import VerificationSummary, aggregate_assessments
from .claim_extractor import Claim, extract_claims
from .evidence_extractor import extract_evidence
from .verifier import verify_claim


@dataclass
class VerificationPipeline:
    """Dependency-injection boundary for the evidence workflow."""

    search_client: SearchClient
    document_fetcher: DocumentFetcher | None = None
    source_policy: SourcePolicy | None = None
    max_claims: int = 5
    max_sources: int = 5
    max_passages: int = 3
    recency_days: int | None = None

    def extract_claims(self, article_text: str) -> list[Claim]:
        return extract_claims(article_text, max_claims=self.max_claims)

    def verify(self, article_text: str) -> VerificationSummary:
        """Extract claims, retrieve sources, and assess each claim."""

        assessments = []
        for claim in self.extract_claims(article_text):
            results = []
            seen_urls: set[str] = set()
            for query in build_queries(claim.text):
                for result in self.search_client.search(
                    query,
                    max_results=self.max_sources,
                    recency_days=self.recency_days,
                ):
                    if result.url in seen_urls:
                        continue
                    seen_urls.add(result.url)
                    results.append(result)

            results = filter_sources(results, self.source_policy)[: self.max_sources]
            documents = self._documents_from_results(results)
            ranked_documents = rank_documents(claim.text, documents)
            passages = extract_evidence(
                claim.text,
                [item.document for item in ranked_documents],
                max_passages=self.max_passages,
            )
            assessments.append(verify_claim(claim, passages))
        return aggregate_assessments(assessments)

    def _documents_from_results(self, results) -> list[RetrievedDocument]:
        documents: list[RetrievedDocument] = []
        for result in results:
            if self.document_fetcher is None:
                if result.snippet:
                    documents.append(
                        RetrievedDocument(
                            url=result.url,
                            title=result.title,
                            text=result.snippet,
                            published_at=(
                                result.published_at.isoformat()
                                if result.published_at
                                else None
                            ),
                            source_name=result.source_name,
                        )
                    )
                continue
            try:
                documents.append(self.document_fetcher.fetch(result.url))
            except (httpx.HTTPError, KeyError, OSError, RuntimeError, ValueError):
                # One unavailable source must not prevent other evidence from
                # being assessed. The failure is represented as insufficient
                # evidence if no other source remains.
                continue
        return documents
