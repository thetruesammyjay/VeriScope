"""Source-policy contracts and filtering helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from urllib.parse import urlparse

from .search_client import SearchResult


@dataclass(frozen=True)
class SourcePolicy:
    """Configuration for allowed and preferred evidence sources."""

    allowed_domains: tuple[str, ...] = ()
    blocked_domains: tuple[str, ...] = ()
    preferred_domains: tuple[str, ...] = ()
    max_age_days: int | None = None
    require_public_url: bool = True
    metadata: dict[str, str] = field(default_factory=dict)


def filter_sources(
    results: list[SearchResult],
    policy: SourcePolicy | None = None,
) -> list[SearchResult]:
    """Apply basic source policy without treating search rank as credibility."""

    if policy is None:
        return list(results)

    filtered: list[SearchResult] = []
    for result in results:
        parsed = urlparse(result.url)
        host = (parsed.hostname or "").lower().rstrip(".")
        if policy.require_public_url and parsed.scheme not in {"http", "https"}:
            continue
        if policy.require_public_url and not host:
            continue
        if any(_matches_domain(host, domain) for domain in policy.blocked_domains):
            continue
        if policy.allowed_domains and not any(
            _matches_domain(host, domain) for domain in policy.allowed_domains
        ):
            continue
        if policy.max_age_days is not None and result.published_at is not None:
            published_at = result.published_at
            if published_at.tzinfo is None:
                published_at = published_at.replace(tzinfo=UTC)
            age = datetime.now(UTC) - published_at
            if age.days > policy.max_age_days:
                continue
        filtered.append(result)
    return filtered


def _matches_domain(host: str, domain: str) -> bool:
    normalized = domain.lower().strip().lstrip(".").rstrip(".")
    return bool(normalized) and (host == normalized or host.endswith(f".{normalized}"))
