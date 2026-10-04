from types import SimpleNamespace

from apps.api.api.dependencies import (
    get_claim_question_service,
    get_deepseek_explainer,
    get_inference_service,
    get_verification_pipeline,
)
from apps.api.core.config import Settings
from apps.api.main import create_app
from apps.api.schemas.question import ClaimQuestionResponse
from apps.api.services.inference_service import InferenceService
from fastapi.testclient import TestClient
from ml.classical.predict import ClassicalPredictor
from ml.classical.train import train_model
from ml.retrieval.search_client import InMemorySearchClient, SearchResult
from ml.verification.pipeline import VerificationPipeline


def test_claim_question_endpoint_is_separate_from_article_analysis():
    class FakeClaimQuestionService:
        def ask(self, question):
            assert question == "Did this news event happen?"
            return ClaimQuestionResponse(
                status="insufficient_evidence",
                message="The retrieved sources don't establish a clear answer.",
            )

    app = create_app()
    app.dependency_overrides[get_claim_question_service] = lambda: FakeClaimQuestionService()
    try:
        client = TestClient(app)
        response = client.post("/api/v1/ask", json={"question": "Did this news event happen?"})
        short_response = client.post("/api/v1/ask", json={"question": "?"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["status"] == "insufficient_evidence"
    assert short_response.status_code == 422


def test_analyze_returns_fixture_based_evidence():
    pipeline = VerificationPipeline(
        search_client=InMemorySearchClient(
            [
                SearchResult(
                    title="Public report",
                    url="https://example.org/report",
                    snippet="The city has 3 hospitals.",
                )
            ]
        )
    )
    app = create_app()
    app.dependency_overrides[get_verification_pipeline] = lambda: pipeline
    app.dependency_overrides[get_deepseek_explainer] = lambda: None
    # Keep this test independent of a locally downloaded production artifact.
    app.dependency_overrides[get_inference_service] = lambda: InferenceService()

    try:
        response = TestClient(app).post(
            "/api/v1/analyze",
            json={
                "text": (
                    "The city has 3 hospitals that provide emergency care, maternity "
                    "services, and routine treatment to residents across the local "
                    "community throughout the year."
                )
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["verification"]["status"] == "supported"
    assert response.json()["verification"]["review_mode"] == "standard"
    assert response.json()["prediction"]["available"] is False
    assert response.json()["verification"]["claims"][0]["evidence"][0]["url"] == (
        "https://example.org/report"
    )


def test_analyze_includes_classical_prediction_when_artifact_is_loaded():
    artifact = train_model(
        [
            "official government report confirms the election result",
            "verified public health announcement from the ministry",
            "secret aliens control the election with invisible machines",
            "shocking miracle cure is hidden by doctors",
        ],
        ["likely_real", "likely_real", "likely_fake", "likely_fake"],
    )
    pipeline = VerificationPipeline(search_client=InMemorySearchClient([]))
    app = create_app()
    app.dependency_overrides[get_verification_pipeline] = lambda: pipeline
    app.dependency_overrides[get_deepseek_explainer] = lambda: None
    app.dependency_overrides[get_inference_service] = lambda: InferenceService(
        ClassicalPredictor(artifact)
    )

    try:
        response = TestClient(app).post(
            "/api/v1/analyze",
            json={
                "text": (
                    "The ministry issued a verified public announcement about the public "
                    "health programme and the new community services available this month."
                )
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["prediction"]["available"] is True
    assert response.json()["prediction"]["label"] in {"likely_real", "likely_fake"}


def test_transformer_prediction_uses_cautious_retrieval_response_only():
    pipeline = VerificationPipeline(
        search_client=InMemorySearchClient(
            [
                SearchResult(
                    title="NASA Perseverance landing",
                    url="https://www.jpl.nasa.gov/mars/perseverance",
                    snippet=(
                        "NASA's Perseverance rover landed on Mars on February 18, "
                        "2021, at Jezero Crater."
                    ),
                )
            ]
        ),
        max_claims=1,
    )
    predictor = SimpleNamespace(
        predict=lambda text: SimpleNamespace(
            label="likely_fake",
            confidence=0.99,
            confidence_method="temperature_scaled",
            model="transformer_sequence_classifier",
            model_version="transformer-test",
            processing_time_ms=5.0,
            disclaimer="Text-pattern output is not a factual verdict.",
        )
    )
    app = create_app()
    app.dependency_overrides[get_verification_pipeline] = lambda: pipeline
    app.dependency_overrides[get_deepseek_explainer] = lambda: _FakeExplainer()
    app.dependency_overrides[get_inference_service] = lambda: InferenceService(predictor)

    try:
        response = TestClient(app).post(
            "/api/v1/analyze",
            json={
                "text": (
                    "NASA's Perseverance rover landed on Mars in February 2021. "
                    "The mission continues to explore Jezero Crater and collect "
                    "rock samples for possible study on Earth."
                )
            },
        )
    finally:
        app.dependency_overrides.clear()

    body = response.json()
    assert response.status_code == 200
    assert body["verification"]["review_mode"] == "transformer_retrieval"
    assert body["verification"]["status"] == "sources_found"
    assert body["verification"]["claims"][0]["status"] == "sources_found"
    assert body["verification"]["claims"][0]["evidence"][0]["source_classification"] == (
        "primary_official"
    )
    assert body["prediction"]["model"] == "transformer_sequence_classifier"
    assert body["prediction"]["confidence"] == 0.99
    assert body["prediction"]["confidence_method"] == "temperature_scaled"
    assert body["explanation"]["available"] is True
    assert body["explanation"]["model"] == "deepseek-flash"
    assert "does not verify" in body["explanation"]["text"]


def test_analyze_enforces_the_active_article_length_limits():
    settings = Settings(
        _env_file=None,
        min_article_length=100,
        max_article_length=120,
    )
    app = create_app(settings)
    app.dependency_overrides[get_deepseek_explainer] = lambda: None
    app.dependency_overrides[get_verification_pipeline] = lambda: VerificationPipeline(
        search_client=InMemorySearchClient([])
    )
    app.dependency_overrides[get_inference_service] = lambda: InferenceService()
    client = TestClient(app)

    try:
        short_response = client.post("/api/v1/analyze", json={"text": "too short"})
        whitespace_response = client.post("/api/v1/analyze", json={"text": " " * 100})
        long_response = client.post("/api/v1/analyze", json={"text": "a" * 121})
        valid_response = client.post("/api/v1/analyze", json={"text": "a" * 100})
    finally:
        app.dependency_overrides.clear()

    assert short_response.status_code == 422
    assert short_response.json()["detail"] == (
        "Article text must contain at least 100 non-whitespace characters."
    )
    assert whitespace_response.status_code == 422
    assert long_response.status_code == 422
    assert long_response.json()["detail"] == "Article text must not exceed 120 characters."
    assert valid_response.status_code == 200


class _FakeExplainer:
    def explain(self, **kwargs):
        del kwargs
        return SimpleNamespace(
            text="The score reflects learned text patterns; it does not verify the claims.",
            model="deepseek-flash",
        )
