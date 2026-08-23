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
