"""Extract passages relevant to a claim from fetched documents."""

from __future__ import annotations

from dataclasses import dataclass

from ml.retrieval.document_fetcher import RetrievedDocument


@dataclass(frozen=True)
class EvidencePassage:
    document_url: str
    text: str
    relevance_score: float = 0.0
    start_offset: int | None = None
    end_offset: int | None = None


def extract_evidence(
    claim: str,
    documents: list[RetrievedDocument],
    *,
    max_passages: int = 5,
) -> list[EvidencePassage]:
    """Return passages that may support or contradict ``claim``."""

    del claim, documents, max_passages
    return []

