import socket

import pytest

from ml.retrieval.document_fetcher import validate_public_url


def test_validate_public_url_accepts_public_http_host(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "ml.retrieval.document_fetcher.socket.getaddrinfo",
        lambda *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))
        ],
    )

    assert validate_public_url("https://example.com/article") == "https://example.com/article"


def test_validate_public_url_rejects_private_address(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "ml.retrieval.document_fetcher.socket.getaddrinfo",
        lambda *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))
        ],
    )

    with pytest.raises(ValueError, match="public addresses"):
        validate_public_url("http://internal.example/article")


def test_validate_public_url_rejects_non_http_scheme():
    with pytest.raises(ValueError, match="HTTP or HTTPS"):
        validate_public_url("file:///etc/passwd")
