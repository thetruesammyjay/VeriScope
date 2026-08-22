"""Source-policy contracts and filtering helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

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
        host = result.url.lower()
        if policy.require_public_url and not result.url.startswith(("http://", "https://")):
            continue
        if any(domain.lower() in host for domain in policy.blocked_domains):
            continue
        if policy.allowed_domains and not any(domain.lower() in host for domain in policy.allowed_domains):
            continue
        filtered.append(result)
    return filtered

