from ml.retrieval.document_fetcher import RetrievedDocument
from ml.verification.evidence_extractor import extract_evidence


def test_extract_evidence_preserves_source_traceability():
    document = RetrievedDocument(
        url="https://example.org/report",
        title="Public report",
        text="The city has 3 hospitals. The report was published yesterday.",
    )

    evidence = extract_evidence("The city has 3 hospitals.", [document])

    assert len(evidence) == 1
    assert evidence[0].document_url == document.url
    assert evidence[0].title == document.title
    assert evidence[0].relevance_score > 0


def test_quality_extraction_filters_weak_passages_and_deduplicates_sources():
    document = RetrievedDocument(
        url="https://science.nasa.gov/mars/landing/?utm_source=test#details",
        title="Perseverance landing",
        text=(
            "NASA's Perseverance rover landed on Mars in February 2021. "
            "Mars exploration continues."
        ),
    )
    weak_document = RetrievedDocument(
        url="https://example.org/unrelated",
        title="Unrelated page",
        text="Mars is a planet.",
    )

    evidence = extract_evidence(
        "NASA's Perseverance rover landed on Mars in February 2021.",
        [document, weak_document],
        min_term_overlap=2,
        min_relevance_score=0.25,
        one_passage_per_source=True,
    )

    assert len(evidence) == 1
    assert evidence[0].document_url == document.url
    assert evidence[0].source_classification == "primary_official"
