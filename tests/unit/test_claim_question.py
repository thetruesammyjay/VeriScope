import json
from datetime import UTC, datetime

import pytest
from apps.api.schemas.question import ClaimQuestionSource
from apps.api.services.claim_question import (
    ClaimAnswer,
    ClaimAnswerError,
    ClaimQuestionService,
    DeepSeekClaimAnswerer,
)
from ml.retrieval.document_fetcher import InMemoryDocumentFetcher, RetrievedDocument
from ml.retrieval.search_client import InMemorySearchClient, SearchResult


class _Answerer:
    def __init__(self, response=None):
        self.response = response or ClaimAnswer(
            answer="The retrieved report says he died in 2025. [S1]",
            source_ids=("S1",),
            sufficient_evidence=True,
        )
        self.called = False

    def answer(self, question, sources):
        self.called = True
        assert question == "Did Muhammadu Buhari die?"
        assert sources[0].source_id == "S1"
        return self.response


def _service(*, results=None, answerer=None):
    source_url = "https://example.org/buhari-report"
    return ClaimQuestionService(
        search_client=InMemorySearchClient(
            results
            or [
                SearchResult(
                    title="Report on Muhammadu Buhari",
                    url=source_url,
                    snippet="Muhammadu Buhari died in London in 2025.",
                    published_at=datetime(2025, 7, 13, tzinfo=UTC),
                    source_name="Example News",
                )
            ]
        ),
        document_fetcher=InMemoryDocumentFetcher(
            {
                source_url: RetrievedDocument(
                    url=source_url,
                    title="Example News report",
                    text=(
                        "Former Nigerian president Muhammadu Buhari died in London "
                        "on July 13, 2025."
                    ),
                    retrieved_at="2025-07-14T00:00:00+00:00",
                )
            }
        ),
        answerer=answerer,
        max_sources=4,
    )


def test_claim_question_returns_grounded_answer_and_source_links():
    answerer = _Answerer()
    response = _service(answerer=answerer).ask("Did Muhammadu Buhari die?")

    assert response.status == "answered"
    assert response.answer == "The retrieved report says he died in 2025. [S1]"
    assert response.cited_source_ids == ["S1"]
    assert response.sources[0].url == "https://example.org/buhari-report"
    assert response.sources[0].published_at == "2025-07-13T00:00:00+00:00"
    assert answerer.called


def test_no_relevant_retrieved_passage_fails_closed_without_calling_deepseek():
    answerer = _Answerer()
    service = _service(
        results=[
            SearchResult(
                title="Local football fixtures",
                url="https://example.org/football",
                snippet="The local team plays its next match on Saturday.",
            )
        ],
        answerer=answerer,
    )

    response = service.ask("Did Muhammadu Buhari die?")

    assert response.status == "insufficient_evidence"
    assert response.sources == []
    assert not answerer.called


def test_deepseek_insufficient_evidence_is_returned_without_a_generated_answer():
    answerer = _Answerer(ClaimAnswer(None, (), False))
    response = _service(answerer=answerer).ask("Did Muhammadu Buhari die?")

    assert response.status == "insufficient_evidence"
    assert response.answer is None
    assert response.sources


def test_deepseek_claim_answerer_limits_input_and_requires_valid_citations(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "sufficient_evidence": True,
                                    "answer": "The report confirms the event. [S1]",
                                    "source_ids": ["S1"],
                                }
                            )
                        }
                    }
                ]
            }

    def fake_post(url, *, headers, json, timeout):
        captured.update(url=url, json=json, timeout=timeout)
        return Response()

    monkeypatch.setattr("apps.api.services.claim_question.httpx.post", fake_post)
    source = ClaimQuestionSource(
        source_id="S1",
        title="Example",
        url="https://example.org",
        excerpt="A" * 100,
        relevance_score=1,
    )
    generated = DeepSeekClaimAnswerer(
        api_key="test-secret", max_input_chars=20, max_output_tokens=96
    ).answer("Did this happen?", [source])

    supplied = json.loads(captured["json"]["messages"][1]["content"])
    assert len(supplied["sources"][0]["excerpt"]) == 20
    assert captured["json"]["max_tokens"] == 96
    assert generated.source_ids == ("S1",)
    assert generated.sufficient_evidence


def test_deepseek_rejects_unrecognized_citation_ids(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "sufficient_evidence": True,
                                    "answer": "The event happened. [S9]",
                                    "source_ids": ["S9"],
                                }
                            )
                        }
                    }
                ]
            }

    monkeypatch.setattr(
        "apps.api.services.claim_question.httpx.post",
        lambda *args, **kwargs: Response(),
    )
    with pytest.raises(ClaimAnswerError):
        DeepSeekClaimAnswerer(api_key="test-secret").answer(
            "Did this happen?",
            [
                ClaimQuestionSource(
                    source_id="S1",
                    title="Example",
                    url="https://example.org",
                    excerpt="A report.",
                    relevance_score=1,
                )
            ],
        )
