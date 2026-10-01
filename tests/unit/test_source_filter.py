from ml.retrieval.search_client import SearchResult
from ml.retrieval.source_filter import (
    SourcePolicy,
    filter_sources,
    preferred_domains_for_context,
)


def test_source_filter_matches_exact_domains_and_subdomains():
    results = [
        SearchResult("Allowed", "https://www.example.org/story"),
        SearchResult("Lookalike", "https://example.org.attacker.test/story"),
        SearchResult("Blocked", "https://spam.test/story"),
    ]

    filtered = filter_sources(
        results,
        SourcePolicy(
            allowed_domains=("example.org",),
            blocked_domains=("spam.test",),
        ),
    )

    assert [result.title for result in filtered] == ["Allowed"]


def test_source_filter_blocks_wikipedia_and_its_subdomains():
    results = [
        SearchResult("Wikipedia", "https://en.wikipedia.org/wiki/Example"),
        SearchResult("Other Wikipedia", "https://de.wikipedia.org/wiki/Example"),
        SearchResult("News outlet", "https://news.example.org/article"),
    ]

    filtered = filter_sources(
        results,
        SourcePolicy(blocked_domains=("wikipedia.org",)),
    )

    assert [result.title for result in filtered] == ["News outlet"]


def test_source_filter_prioritizes_preferred_domain_without_dropping_others():
    results = [
        SearchResult("Other coverage", "https://example.org/nasa-story"),
        SearchResult("NASA report", "https://science.nasa.gov/perseverance"),
    ]

    filtered = filter_sources(
        results,
        SourcePolicy(preferred_domains=preferred_domains_for_context("NASA Perseverance")),
    )

    assert [result.title for result in filtered] == ["NASA report", "Other coverage"]
