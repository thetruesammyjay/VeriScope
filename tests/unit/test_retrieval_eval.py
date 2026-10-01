import json
from pathlib import Path

from ml.evaluation.retrieval_metrics import retrieval_metrics
from scripts.evaluate_retrieval_fixtures import evaluate_fixture

FIXTURES = Path(__file__).parents[1] / "fixtures/evidence_retrieval_eval.json"


def test_retrieval_metrics_deduplicate_urls_and_calculate_precision_recall():
    metrics = retrieval_metrics(
        ["https://www.example.org/page?utm_source=x", "https://example.org/page"],
        ["https://example.org/page"],
        k=1,
    )

    assert metrics["claim_coverage"] == 1.0
    assert metrics["evidence_precision_at_k"] == 1.0
    assert metrics["evidence_recall_at_k"] == 1.0
    assert metrics["source_diversity"] == 1.0


def test_retrieval_fixture_records_nasa_and_nigerian_primary_pages():
    fixture = json.loads(FIXTURES.read_text(encoding="utf-8"))
    cases = {case["id"]: case for case in fixture["cases"]}

    assert any(
        source["relevant"] and source["source_classification"] == "primary_official"
        for source in cases["nasa-perseverance-landing"]["sources"]
    )
    assert any(
        source["relevant"] and source["source_classification"] == "primary_official"
        for source in cases["nbs-june-2024-inflation"]["sources"]
    )


def test_fixture_retrieval_report_keeps_good_coverage_and_filters_weak_results():
    report = evaluate_fixture()

    assert report["sample_count"] == 5
    assert report["summary"]["claim_coverage"] >= 0.8
    assert report["summary"]["evidence_precision_at_k"] >= 0.4
