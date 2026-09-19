from datetime import datetime

import pytest

from apps.api.api.dependencies import build_search_client
from apps.api.core.config import Settings
from ml.retrieval.search_client import BraveSearchClient, EmptySearchClient


class _FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self):
        return {
            "web": {
                "results": [
                    {
                        "title": "Official report",
                        "url": "https://example.org/report",
                        "description": "Published figures from the official report.",
                        "page_age": "2026-01-10T12:00:00+00:00",
                        "profile": {"long_name": "Example Organisation"},
                    }
                ]
            }
        }


def test_brave_search_maps_web_results_and_authentication(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_get(url, *, params, headers, timeout):
        captured.update(url=url, params=params, headers=headers, timeout=timeout)
        return _FakeResponse()

    monkeypatch.setattr("ml.retrieval.search_client.httpx.get", fake_get)
    results = BraveSearchClient(api_key="secret", timeout_seconds=12).search(
        "official figures", max_results=8, recency_days=7
    )

    assert captured["url"] == "https://api.search.brave.com/res/v1/web/search"
    assert captured["params"] == {"q": "official figures", "count": 8, "freshness": "pw"}
    assert captured["headers"]["X-Subscription-Token"] == "secret"
    assert captured["timeout"] == 12
    assert results[0].title == "Official report"
    assert results[0].source_name == "Example Organisation"
    assert results[0].published_at == datetime.fromisoformat("2026-01-10T12:00:00+00:00")


def test_build_search_client_selects_brave_with_an_api_key():
    client = build_search_client(
        Settings(_env_file=None, search_provider="brave", search_api_key="secret")
    )

    assert isinstance(client, BraveSearchClient)


def test_build_search_client_defaults_to_empty_without_a_key():
    client = build_search_client(Settings(_env_file=None, search_provider="brave"))

    assert isinstance(client, EmptySearchClient)
