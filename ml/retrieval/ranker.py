"""Evidence candidate ranking contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .document_fetcher import RetrievedDocument


@dataclass(frozen=True)
class RankedDocument:
    document: RetrievedDocument
    relevance_score: float
    source_quality_score: float | None = None


def rank_documents(
    query: str,
    documents: list[RetrievedDocument],
) -> list[RankedDocument]:
    """Rank documents using deterministic lexical overlap.

    This is deliberately a transparent baseline. A learned ranker can replace
    it later without changing the retrieval contract.
    """

    query_terms = _terms(query)
    ranked = []
    for document in documents:
        document_terms = _terms(document.text)
        overlap = len(query_terms & document_terms)
        score = overlap / len(query_terms) if query_terms else 0.0
        ranked.append(RankedDocument(document=document, relevance_score=score))
    return sorted(ranked, key=lambda item: item.relevance_score, reverse=True)


def _terms(text: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[a-z0-9]{3,}", text.lower())
        if term not in {"the", "and", "for", "that", "with", "this", "from"}
    }
