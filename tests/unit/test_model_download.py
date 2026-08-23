import hashlib
from pathlib import Path

import pytest

from scripts.download_model import download_artifact


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self, size: int = -1) -> bytes:
        if not self._payload:
            return b""
        chunk, self._payload = self._payload[:size], self._payload[size:]
        return chunk


def test_download_artifact_writes_and_verifies_atomically(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    payload = b"model artifact"
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(
        "scripts.download_model.urlopen", lambda request, timeout: _FakeResponse(payload)
    )

    destination = tmp_path / "models" / "model.joblib"
    result = download_artifact(
        "https://github.com/example/model.joblib",
        destination,
        sha256=digest,
    )

    assert result == destination
    assert destination.read_bytes() == payload
    assert list(destination.parent.glob("*.tmp")) == []


def test_download_artifact_rejects_invalid_checksum(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(
        "scripts.download_model.urlopen", lambda request, timeout: _FakeResponse(b"model")
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        download_artifact(
            "https://github.com/example/model.joblib",
            tmp_path / "model.joblib",
            sha256="0" * 64,
        )

    assert not (tmp_path / "model.joblib").exists()


def test_download_artifact_reuses_matching_existing_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    destination = tmp_path / "model.joblib"
    destination.write_bytes(b"existing model")
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    monkeypatch.setattr(
        "scripts.download_model.urlopen",
        lambda request, timeout: pytest.fail("existing artifact should be reused"),
    )

    assert download_artifact(
        "https://github.com/example/model.joblib",
        destination,
        sha256=digest,
    ) == destination
