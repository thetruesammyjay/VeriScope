from ml.retrieval.search_client import SearchResult
from ml.retrieval.source_filter import SourcePolicy, filter_sources


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
