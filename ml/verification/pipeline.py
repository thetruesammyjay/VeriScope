"""Orchestrate claim extraction, retrieval, and evidence verification."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

from ml.retrieval.document_fetcher import DocumentFetcher, RetrievedDocument
from ml.retrieval.query_builder import build_queries
from ml.retrieval.ranker import rank_documents
from ml.retrieval.search_client import SearchClient
from ml.retrieval.source_filter import (
    SourcePolicy,
    canonical_source_url,
    filter_sources,
    preferred_domains_for_context,
)

from .aggregator import VerificationSummary, aggregate_assessments
from .claim_extractor import Claim, extract_claims
from .evidence_extractor import extract_evidence
from .verifier import ClaimAssessment, verify_claim

logger = logging.getLogger(__name__)


@dataclass
class VerificationPipeline:
    """Dependency-injection boundary for the evidence workflow."""

    search_client: SearchClient
    document_fetcher: DocumentFetcher | None = None
    source_policy: SourcePolicy | None = None
    max_search_results: int = 10
    max_claims: int = 5
    max_sources: int = 5
    max_passages: int = 3
    recency_days: int | None = None

    def extract_claims(self, article_text: str) -> list[Claim]:
        return extract_claims(article_text, max_claims=self.max_claims)

    def verify(
        self,
        article_text: str,
        *,
        transformer_retrieval: bool = False,
    ) -> VerificationSummary:
        """Extract claims, retrieve sources, and assess each claim."""

        assessments = []
        search_timed_out = False
        for claim in self.extract_claims(article_text):
            results = []
            seen_urls: set[str] = set()
            queries = build_queries(
                claim.text,
                context=article_text if transformer_retrieval else None,
            )
            for query in queries:
                if search_timed_out:
                    break
                try:
                    search_results = self.search_client.search(
                        query,
                        max_results=self.max_search_results,
                        recency_days=self.recency_days,
                    )
                except httpx.TimeoutException:
                    # A slow provider should reduce evidence availability, not
                    # turn a valid analysis request into an API 500. Stop
                    # further searches for this request to avoid compounding
                    # latency or provider usage.
                    logger.warning("Search provider timed out; returning available evidence")
                    search_timed_out = True
                    break
                for result in search_results:
                    canonical_url = canonical_source_url(result.url)
                    if canonical_url in seen_urls:
                        continue
                    seen_urls.add(canonical_url)
                    results.append(result)

            policy = self.source_policy
            if transformer_retrieval:
                base = policy or SourcePolicy()
                policy = SourcePolicy(
                    allowed_domains=base.allowed_domains,
                    blocked_domains=base.blocked_domains,
                    preferred_domains=tuple(
                        dict.fromkeys(
                            (*base.preferred_domains, *preferred_domains_for_context(article_text))
                        )
                    ),
                    max_age_days=base.max_age_days,
                    require_public_url=base.require_public_url,
                    metadata=base.metadata,
                )
            results = filter_sources(results, policy)
            if transformer_retrieval:
                results = _rank_search_results(claim.text, results)
                results = _one_result_per_domain(results)
            results = results[: self.max_sources]
            documents = self._documents_from_results(results)
            ranked_documents = rank_documents(claim.text, documents)
            passages = extract_evidence(
                claim.text,
                [item.document for item in ranked_documents],
                max_passages=self.max_passages,
                min_term_overlap=2 if transformer_retrieval else 1,
                min_relevance_score=0.25 if transformer_retrieval else 0.0,
                one_passage_per_source=transformer_retrieval,
            )
            if transformer_retrieval:
                assessments.append(
                    ClaimAssessment(
                        claim=claim,
                        status="sources_found" if passages else "insufficient",
                        evidence=tuple(passages),
                        rationale=(
                            "Retrieved passages have lexical overlap with this claim; "
                            "this is not a determination that the claim is true."
                            if passages
                            else "No sufficiently relevant source passage was retrieved."
                        ),
                    )
                )
            else:
                assessments.append(verify_claim(claim, passages))
        return aggregate_assessments(
            assessments,
            transformer_retrieval=transformer_retrieval,
        )

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


def _rank_search_results(claim: str, results):
    snippets = [
        RetrievedDocument(
            url=result.url,
            title=result.title,
            text=f"{result.title} {result.snippet}",
            source_name=result.source_name,
        )
        for result in results
    ]
    ranked = rank_documents(claim, snippets)
    by_url = {canonical_source_url(result.url): result for result in results}
    return [by_url[canonical_source_url(item.document.url)] for item in ranked]


def _one_result_per_domain(results):
    selected = []
    seen_domains: set[str] = set()
    for result in results:
        domain = (urlsplit(result.url).hostname or "").lower().removeprefix("www.")
        if domain in seen_domains:
            continue
        seen_domains.add(domain)
        selected.append(result)
    return selected
