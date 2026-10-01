"""Evaluate the transformer retrieval baseline against hand-labelled fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from urllib.parse import urlsplit

from ml.evaluation.retrieval_metrics import retrieval_metrics
from ml.retrieval.document_fetcher import RetrievedDocument
from ml.verification.evidence_extractor import extract_evidence

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "tests/fixtures/evidence_retrieval_eval.json"
OUTPUT_PATH = ROOT / "reports/metrics/transformer_retrieval/fixture_metrics.json"


def evaluate_fixture(path: Path = FIXTURE_PATH) -> dict:
    """Run lexical relevance selection and compare it with labelled source URLs."""

    fixture = json.loads(path.read_text(encoding="utf-8"))
    case_results = []
    for case in fixture["cases"]:
        documents = [
            RetrievedDocument(
                url=source["url"],
                title=source["title"],
                text=source["text"],
                source_name=(urlsplit(source["url"]).hostname or ""),
            )
            for source in case["sources"]
        ]
        passages = extract_evidence(
            case["claim"],
            documents,
            max_passages=10,
            min_term_overlap=2,
            min_relevance_score=0.25,
            one_passage_per_source=True,
        )
        relevant_urls = [source["url"] for source in case["sources"] if source["relevant"]]
        metrics = retrieval_metrics(
            [passage.document_url for passage in passages],
            relevant_urls,
            k=5,
        )
        case_results.append(
            {
                "id": case["id"],
                "claim": case["claim"],
                "relevant_urls": relevant_urls,
                "retrieved_urls": [passage.document_url for passage in passages],
                "metrics": metrics,
            }
        )

    metric_names = (
        "claim_coverage",
        "evidence_precision_at_k",
        "evidence_recall_at_k",
        "source_diversity",
    )
    summary = {
        name: mean(case["metrics"][name] for case in case_results)
        for name in metric_names
    }
    return {
        "evaluation_type": "small_hand_labelled_retrieval_diagnostic",
        "sample_count": len(case_results),
        "selection_standard": {
            "minimum_distinct_term_overlap": 2,
            "minimum_claim_term_relevance": 0.25,
            "one_passage_per_canonical_url": True,
        },
        "summary": summary,
        "cases": case_results,
    }


def main() -> None:
    report = evaluate_fixture()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Saved retrieval evaluation to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()


__all__ = ["evaluate_fixture", "main"]
