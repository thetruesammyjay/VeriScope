"""Evidence candidate ranking contracts."""

from __future__ import annotations

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
    """Return document candidates in deterministic placeholder order.

    Semantic ranking and source-quality scoring will be implemented after the
    search provider and verification evaluation protocol are selected.
    """

    del query
    return [RankedDocument(document=document, relevance_score=0.0) for document in documents]

