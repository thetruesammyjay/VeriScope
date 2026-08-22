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
