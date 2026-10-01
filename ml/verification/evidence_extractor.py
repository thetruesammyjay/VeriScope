"""Extract passages relevant to a claim from fetched documents."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ml.retrieval.document_fetcher import RetrievedDocument
from ml.retrieval.source_filter import canonical_source_url, is_primary_source


@dataclass(frozen=True)
class EvidencePassage:
    document_url: str
    text: str
    relevance_score: float = 0.0
    start_offset: int | None = None
    end_offset: int | None = None
    title: str | None = None
    source_name: str | None = None
    source_classification: str = "unclassified"
    published_at: str | None = None
    retrieved_at: str | None = None


def extract_evidence(
    claim: str,
    documents: list[RetrievedDocument],
    *,
    max_passages: int = 5,
    min_term_overlap: int = 1,
    min_relevance_score: float = 0.0,
    one_passage_per_source: bool = False,
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
            if overlap < min_term_overlap or score < min_relevance_score:
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
                    source_classification=(
                        "primary_official"
                        if is_primary_source(document.url)
                        else "unclassified"
                    ),
                    published_at=document.published_at,
                    retrieved_at=document.retrieved_at,
                )
            )
    candidates.sort(
        key=lambda passage: (
            passage.relevance_score,
            passage.source_classification == "primary_official",
        ),
        reverse=True,
    )
    if one_passage_per_source:
        unique: dict[str, EvidencePassage] = {}
        for passage in candidates:
            unique.setdefault(canonical_source_url(passage.document_url), passage)
        candidates = list(unique.values())
    return candidates[:max_passages]


def _terms(text: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[a-z0-9]{3,}", text.lower())
        if term not in {
            "the", "and", "for", "that", "with", "this", "from", "are",
            "was", "were", "has", "have", "had", "its", "their", "they",
            "them", "into", "over", "under", "about", "after", "before",
            "than", "then", "when", "where", "what", "which", "while",
            "also", "according", "said", "says", "report", "reports", "claim",
            "claims", "article", "articles", "news", "statement", "statements",
        }
    }


def _sentences(text: str) -> list[tuple[int, int, str]]:
    result: list[tuple[int, int, str]] = []
    for match in re.finditer(r"[^.!?]+(?:[.!?]+|$)", text, flags=re.DOTALL):
        sentence = " ".join(match.group(0).split())
        if sentence:
            result.append((match.start(), match.end(), sentence))
    return result
