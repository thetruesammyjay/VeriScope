from fastapi.testclient import TestClient

from apps.api.api.dependencies import get_verification_pipeline
from apps.api.main import create_app
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

    try:
        response = TestClient(app).post(
            "/api/v1/analyze",
            json={"text": "The city has 3 hospitals."},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["status"] == "supported"
    assert response.json()["claims"][0]["evidence"][0]["url"] == (
        "https://example.org/report"
    )
