"""Build focused search queries from extracted claims."""

from __future__ import annotations


def build_queries(claim: str, *, max_queries: int = 3) -> list[str]:
    """Return search-query candidates for ``claim``.

    Query generation is intentionally left provider/model agnostic. The
    implementation will later add entity extraction, date constraints, and
    source-specific query variants.
    """

    cleaned = " ".join(claim.split())
    if not cleaned:
        return []
    return [cleaned][:max_queries]

