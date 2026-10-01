"""Tests for deterministic and focused evidence-search queries."""

from ml.retrieval.query_builder import build_queries, context_terms


def test_contextual_query_adds_article_entities_to_claim_once():
    claim = "These reports were released last month."
    article = "Nigeria's National Bureau of Statistics publishes CPI reports. " + claim

    query = build_queries(claim, context=article)[0]

    assert "Nigeria" in query
    assert "Statistics" in query
    assert len(build_queries(claim, context=article)) == 1


def test_context_terms_does_not_repeat_terms_already_in_claim():
    assert context_terms(
        "NASA's Perseverance rover landed in Jezero Crater.",
        "NASA's Perseverance rover landed on Mars.",
    ) == ["Jezero", "Crater"]
