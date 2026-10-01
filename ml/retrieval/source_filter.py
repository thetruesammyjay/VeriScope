"""Source-policy contracts and filtering helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from urllib.parse import parse_qsl, urlencode, urlparse, urlsplit, urlunsplit

from .search_client import SearchResult

PRIMARY_SOURCE_DOMAINS = {
    "nasa.gov",
    "who.int",
    "nigerianstat.gov.ng",
}


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
    if policy.preferred_domains:
        preferred = tuple(policy.preferred_domains)
        filtered.sort(
            key=lambda result: not any(
                _matches_domain((urlparse(result.url).hostname or "").lower(), domain)
                for domain in preferred
            )
        )
    return filtered


def is_primary_source(url: str) -> bool:
    """Identify configured institutional domains, without asserting claim truth."""

    host = (urlparse(url).hostname or "").lower().rstrip(".")
    return any(_matches_domain(host, domain) for domain in PRIMARY_SOURCE_DOMAINS)


def canonical_source_url(url: str) -> str:
    """Normalize URL variants used to deduplicate source pages."""

    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower().removeprefix("www.").rstrip(".")
    path = parsed.path.rstrip("/") or "/"
    ignored = {"fbclid", "gclid", "mc_cid", "mc_eid"}
    query = urlencode(
        sorted(
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if not key.lower().startswith("utm_") and key.lower() not in ignored
        )
    )
    return urlunsplit(("https", host, path, query, ""))


def preferred_domains_for_context(text: str) -> tuple[str, ...]:
    """Return known primary-source domains relevant to prominent context entities."""

    domains: list[str] = []
    if re.search(r"\b(nasa|perseverance|jezero|mars)\b", text, re.IGNORECASE):
        domains.append("nasa.gov")
    if re.search(r"\b(who|covid(?:-19)?)\b", text, re.IGNORECASE):
        domains.append("who.int")
    if re.search(
        r"\b(nigeria|nbs|national bureau of statistics|consumer price index|cpi)\b",
        text,
        re.IGNORECASE,
    ):
        domains.append("nigerianstat.gov.ng")
    return tuple(domains)


def _matches_domain(host: str, domain: str) -> bool:
    normalized = domain.lower().strip().lstrip(".").rstrip(".")
    return bool(normalized) and (host == normalized or host.endswith(f".{normalized}"))


__all__ = [
    "PRIMARY_SOURCE_DOMAINS",
    "SourcePolicy",
    "canonical_source_url",
    "filter_sources",
    "is_primary_source",
    "preferred_domains_for_context",
]
