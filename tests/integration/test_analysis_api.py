from fastapi.testclient import TestClient

from apps.api.api.dependencies import get_inference_service, get_verification_pipeline
from apps.api.main import create_app
from apps.api.services.inference_service import InferenceService
from ml.classical.predict import ClassicalPredictor
from ml.classical.train import train_model
from ml.retrieval.search_client import InMemorySearchClient, SearchResult
from ml.verification.pipeline import VerificationPipeline


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
    # Keep this test independent of a locally downloaded production artifact.
    app.dependency_overrides[get_inference_service] = lambda: InferenceService()

    try:
        response = TestClient(app).post(
            "/api/v1/analyze",
            json={"text": "The city has 3 hospitals."},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["verification"]["status"] == "supported"
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
    app.dependency_overrides[get_inference_service] = lambda: InferenceService(
        ClassicalPredictor(artifact)
    )

    try:
        response = TestClient(app).post(
            "/api/v1/analyze",
            json={"text": "The ministry issued a verified public announcement."},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["prediction"]["available"] is True
    assert response.json()["prediction"]["label"] in {"likely_real", "likely_fake"}
