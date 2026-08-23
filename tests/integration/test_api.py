from fastapi.testclient import TestClient

from apps.api.core.config import Settings
from apps.api.main import create_app


def test_api_starts_and_reports_health():
    settings = Settings(
        _env_file=None,
        app_env="testing",
        model_name="test-model",
        cors_origins="https://web.example",
    )
    client = TestClient(create_app(settings))

    response = client.get(
        "/health",
        headers={"Origin": "https://web.example"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "environment": "testing",
        "model_name": "test-model",
    }
    assert response.headers["access-control-allow-origin"] == "https://web.example"
