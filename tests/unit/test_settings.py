import pytest
from pydantic import ValidationError

from apps.api.core.config import Settings


def test_settings_load_retrieval_environment(monkeypatch):
    monkeypatch.setenv("SEARCH_PROVIDER", "fixture")
    monkeypatch.setenv("SEARCH_MAX_RESULTS", "20")
    monkeypatch.setenv("EVIDENCE_RECENCY_DAYS", "7")

    settings = Settings(_env_file=None)

    assert settings.search_provider == "fixture"
    assert settings.search_max_results == 20
    assert settings.evidence_recency_days == 7


def test_settings_support_render_port_and_cors_origins(monkeypatch):
    monkeypatch.setenv("PORT", "10000")
    monkeypatch.setenv("CORS_ORIGINS", "https://news.example, https://admin.example/")

    settings = Settings(_env_file=None)

    assert settings.api_port == 10000
    assert settings.cors_origin_list == [
        "https://news.example",
        "https://admin.example",
    ]


def test_settings_load_model_artifact_environment(monkeypatch):
    monkeypatch.setenv(
        "MODEL_ARTIFACT_URL",
        "https://github.com/example/repo/releases/download/model-v1/model.joblib",
    )
    monkeypatch.setenv("MODEL_ARTIFACT_SHA256", "a" * 64)
    monkeypatch.setenv("MODEL_ARTIFACT_TOKEN", "secret")

    settings = Settings(_env_file=None)

    assert settings.model_artifact_url.endswith("/model.joblib")
    assert settings.model_artifact_sha256 == "a" * 64
    assert settings.model_artifact_token.get_secret_value() == "secret"


def test_settings_load_transformer_deployment_environment(monkeypatch):
    monkeypatch.setenv("PRODUCTION_MODEL", "transformer")
    monkeypatch.setenv("TRANSFORMER_MODEL_PATH", "models/transformer/release")
    monkeypatch.setenv(
        "TRANSFORMER_MODEL_ARTIFACT_URL",
        "https://github.com/example/repo/releases/download/model-v1/distilbert.tar.gz",
    )
    monkeypatch.setenv("TRANSFORMER_MODEL_ARTIFACT_SHA256", "b" * 64)

    settings = Settings(_env_file=None)

    assert settings.production_model == "transformer"
    assert settings.transformer_model_path.as_posix().endswith("transformer/release")
    assert settings.transformer_model_artifact_url.endswith("distilbert.tar.gz")
    assert settings.transformer_model_artifact_sha256 == "b" * 64


def test_settings_reject_invalid_article_length_range():
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            min_article_length=500,
            max_article_length=100,
        )


def test_settings_reject_invalid_retrieval_limits():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, search_timeout_seconds=0)
