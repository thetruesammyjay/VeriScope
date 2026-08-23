"""Extract passages relevant to a claim from fetched documents."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ml.retrieval.document_fetcher import RetrievedDocument


@dataclass(frozen=True)
class EvidencePassage:
    document_url: str
    text: str
    relevance_score: float = 0.0
    start_offset: int | None = None
    end_offset: int | None = None
    title: str | None = None
    source_name: str | None = None
    published_at: str | None = None
    retrieved_at: str | None = None


def extract_evidence(
    claim: str,
    documents: list[RetrievedDocument],
    *,
    max_passages: int = 5,
) -> list[EvidencePassage]:
    """Return sentence passages with lexical overlap to ``claim``."""

    if max_passages <= 0:
        return []
    claim_terms = _terms(claim)
    if not claim_terms:
        return []

    candidates: list[EvidencePassage] = []
    for document in documents:
        for start, end, sentence in _sentences(document.text):
            passage_terms = _terms(sentence)
            overlap = len(claim_terms & passage_terms)
            score = overlap / len(claim_terms)
            if overlap == 0:
                continue
            candidates.append(
                EvidencePassage(
                    document_url=document.url,
                    text=sentence,
                    relevance_score=score,
                    start_offset=start,
                    end_offset=end,
                    title=document.title,
                    source_name=document.source_name,
                    published_at=document.published_at,
                    retrieved_at=document.retrieved_at,
                )
            )
    candidates.sort(key=lambda passage: passage.relevance_score, reverse=True)
    return candidates[:max_passages]


def _terms(text: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[a-z0-9]{3,}", text.lower())
        if term not in {"the", "and", "for", "that", "with", "this", "from"}
    }


def _sentences(text: str) -> list[tuple[int, int, str]]:
    result: list[tuple[int, int, str]] = []
    for match in re.finditer(r"[^.!?]+(?:[.!?]+|$)", text, flags=re.DOTALL):
        sentence = " ".join(match.group(0).split())
        if sentence:
            result.append((match.start(), match.end(), sentence))
    return result
