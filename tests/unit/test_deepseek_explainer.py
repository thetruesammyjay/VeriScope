import json

import httpx
import pytest
from apps.api.api.dependencies import build_deepseek_explainer
from apps.api.core.config import Settings
from apps.api.services.deepseek_explainer import (
    DEEPSEEK_CHAT_COMPLETIONS_URL,
    DeepSeekExplainer,
    DeepSeekExplanationError,
)


class _FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "choices": [
                {"message": {"content": "This is a cautious explanation."}}
            ]
        }


def test_deepseek_explainer_sends_only_compact_result_metadata(monkeypatch):
    captured = {}

    def fake_post(url, *, headers, json, timeout):
        captured.update(url=url, headers=headers, json=json, timeout=timeout)
        return _FakeResponse()

    monkeypatch.setattr("apps.api.services.deepseek_explainer.httpx.post", fake_post)
    explainer = DeepSeekExplainer(
        api_key="test-secret",
        timeout_seconds=9,
        max_input_chars=30,
        max_output_tokens=100,
    )

    generated = explainer.explain(
        label="likely_fake",
        confidence=0.8132,
        confidence_method="temperature_scaled",
        verification_status="sources_found",
        article_text="Mars exploration details " * 200,
    )

    assert captured["url"] == DEEPSEEK_CHAT_COMPLETIONS_URL
    assert captured["headers"]["Authorization"] == "Bearer test-secret"
    assert captured["timeout"] == 9
    assert captured["json"]["max_tokens"] == 100
    assert captured["json"]["thinking"] == {"type": "disabled"}
    assert captured["json"]["stream"] is False
    assert len(captured["json"]["messages"]) == 2
    user_message = json.loads(captured["json"]["messages"][1]["content"])
    assert user_message["classifier_label"] == "likely_fake"
    assert "0.8132" in user_message["classifier_score"]
    assert user_message["source_review_status"] == "sources_found"
    assert len(user_message["article_excerpt"]) == 30
    assert generated.text == "This is a cautious explanation."
    assert generated.model == "deepseek-flash"


def test_deepseek_explainer_rejects_provider_failures(monkeypatch):
    def fail_post(*args, **kwargs):
        del args, kwargs
        raise httpx.ReadTimeout("provider timed out")

    monkeypatch.setattr("apps.api.services.deepseek_explainer.httpx.post", fail_post)
    explainer = DeepSeekExplainer(api_key="test-secret")

    with pytest.raises(DeepSeekExplanationError):
        explainer.explain(
            label="likely_real",
            confidence=0.8,
            confidence_method="temperature_scaled",
            verification_status="insufficient",
            article_text="Example article text.",
        )


def test_deepseek_explainer_is_disabled_without_a_nonempty_key():
    assert build_deepseek_explainer(Settings(_env_file=None)) is None
    assert build_deepseek_explainer(
        Settings(_env_file=None, deepseek_api_key="")
    ) is None


def test_settings_load_deepseek_budget_controls():
    settings = Settings(
        _env_file=None,
        deepseek_api_key="secret",
        deepseek_model="deepseek-flash",
        deepseek_timeout_seconds=8,
        deepseek_max_input_chars=1800,
        deepseek_max_output_tokens=120,
    )

    assert settings.deepseek_api_key.get_secret_value() == "secret"
    assert settings.deepseek_model == "deepseek-flash"
    assert settings.deepseek_timeout_seconds == 8
    assert settings.deepseek_max_input_chars == 1800
    assert settings.deepseek_max_output_tokens == 120
