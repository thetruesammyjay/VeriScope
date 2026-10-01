import httpx

from ml.retrieval.search_client import SearchResult
from ml.verification.pipeline import VerificationPipeline


class RecordingSearchClient:
    def __init__(self, results):
        self.results = results
        self.queries = []

    def search(self, query, *, max_results=10, recency_days=None):
        self.queries.append(query)
        return self.results[:max_results]


def test_transformer_retrieval_uses_context_deduplicates_and_prefers_primary():
    primary_url = "https://www.jpl.nasa.gov/mars/perseverance?utm_source=brave"
    primary = SearchResult(
        title="NASA confirms Perseverance landing",
        url=primary_url,
        snippet=(
            "NASA's Perseverance rover landed on Mars on February 18, 2021, "
            "and touched down inside Jezero Crater."
        ),
    )
    search = RecordingSearchClient(
        [
            SearchResult(
                title="Other coverage",
                url="https://parkingday.org/nasa-rover-story",
                snippet=(
                    "NASA's Perseverance rover landed on Mars on February 18, 2021."
                ),
            ),
            primary,
            SearchResult(
                title="Duplicate NASA coverage",
                url="https://jpl.nasa.gov/mars/perseverance#landing",
                snippet=primary.snippet,
            ),
            SearchResult(
                title="Unrelated source",
                url="https://www.clevelandfed.org/inflation",
                snippet="United States inflation is estimated with economic indicators.",
            ),
        ]
    )
    pipeline = VerificationPipeline(search_client=search, max_claims=1, max_sources=5)
    article = (
        "NASA's Perseverance rover landed on Mars in February 2021. "
        "The rover is exploring Jezero Crater, a site selected for its ancient lake geology."
    )

    summary = pipeline.verify(article, transformer_retrieval=True)

    assert len(search.queries) == 1
    assert "Jezero" in search.queries[0]
    assert summary.status == "sources_found"
    evidence = summary.assessments[0].evidence
    assert len(evidence) == 2
    assert evidence[0].source_classification == "primary_official"
    assert "jpl.nasa.gov" in evidence[0].document_url
    assert all(item.document_url != "https://www.clevelandfed.org/inflation" for item in evidence)


def test_transformer_retrieval_returns_insufficient_for_weak_overlap():
    pipeline = VerificationPipeline(
        search_client=RecordingSearchClient(
            [
                SearchResult(
                    title="General space story",
                    url="https://example.org/space",
                    snippet="Mars is a planet with a thin atmosphere.",
                )
            ],
        ),
        max_claims=1,
    )

    summary = pipeline.verify(
        "NASA's Perseverance rover landed on Mars on February 18, 2021.",
        transformer_retrieval=True,
    )

    assert summary.status == "insufficient"
    assert summary.assessments[0].evidence == ()


def test_transformer_retrieval_handles_search_timeout_as_insufficient():
    class TimeoutSearchClient:
        calls = 0

        def search(self, query, *, max_results=10, recency_days=None):
            self.calls += 1
            raise httpx.ReadTimeout("search timed out")

    search = TimeoutSearchClient()
    pipeline = VerificationPipeline(search_client=search, max_claims=3)

    summary = pipeline.verify(
        "NASA's rover landed on Mars in 2021. The mission is collecting samples.",
        transformer_retrieval=True,
    )

    assert search.calls == 1
    assert summary.status == "insufficient"
    assert summary.assessments
    assert all(assessment.status == "insufficient" for assessment in summary.assessments)
    assert all(assessment.evidence == () for assessment in summary.assessments)
