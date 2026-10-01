"""Build focused search queries from extracted claims."""

from __future__ import annotations

import re

_CONTEXT_STOPWORDS = {
    "the", "this", "that", "these", "those", "according", "however",
    "therefore", "meanwhile", "furthermore", "february", "january", "march",
    "april", "may", "june", "july", "august", "september", "october",
    "november", "december",
}


def context_terms(article_text: str, claim: str, *, limit: int = 6) -> list[str]:
    """Extract a few names and acronyms from the article for claim disambiguation."""

    claim_terms = {term.lower() for term in re.findall(r"[A-Za-z0-9]+", claim)}
    found: list[str] = []
    seen: set[str] = set()
    # Acronyms often carry more search value than generic words (NASA, WHO, NBS).
    candidates = re.findall(r"\b[A-Z][A-Z0-9]{1,}\b", article_text)
    # Include proper-name words and short phrases, but discard sentence starters.
    candidates.extend(
        match.group(0)
        for match in re.finditer(
            r"\b[A-Z][a-zA-Z0-9'\u2019-]*(?:\s+(?:of|the|and)\s+[A-Z][a-zA-Z0-9'\u2019-]*|\s+[A-Z][a-zA-Z0-9'\u2019-]*){0,3}",
            article_text,
        )
    )
    for candidate in candidates:
        for word in re.findall(r"[A-Za-z0-9]+(?:['\u2019][A-Za-z]+)?", candidate):
            normalized = word.strip("'\u2019").lower()
            if normalized.endswith("'s") or normalized.endswith("\u2019s"):
                normalized = normalized[:-2]
            if (
                len(normalized) < 3
                or normalized in _CONTEXT_STOPWORDS
                or normalized in claim_terms
                or normalized in seen
            ):
                continue
            seen.add(normalized)
            found.append(word)
            if len(found) >= limit:
                return found
    return found


def build_queries(
    claim: str,
    *,
    context: str | None = None,
    max_queries: int = 3,
) -> list[str]:
    """Return one provider-neutral query, enriched with article context when given.

    Keeping this to one query per claim avoids multiplying live search-provider
    requests while adding article entities that disambiguate short claims.
    """

    cleaned = " ".join(claim.split())
    if not cleaned:
        return []
    if context is not None:
        anchors = context_terms(context, cleaned)
        if anchors:
            cleaned = f"{cleaned} {' '.join(anchors)}"
    return [cleaned][:max_queries]


__all__ = ["build_queries", "context_terms"]
