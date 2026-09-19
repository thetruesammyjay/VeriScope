import tarfile
from pathlib import Path

import pytest

from scripts.package_transformer import package_transformer


def _create_transformer_directory(path: Path) -> None:
    path.mkdir()
    for name, content in {
        "config.json": "{}",
        "metadata.json": "{}",
        "tokenizer.json": "{}",
        "model.safetensors": "weights",
    }.items():
        path.joinpath(name).write_text(content, encoding="utf-8")


def test_package_transformer_creates_deterministic_release_asset(tmp_path: Path):
    source = tmp_path / "distilbert"
    _create_transformer_directory(source)
    first = package_transformer(source, tmp_path / "first.tar.gz")
    second = package_transformer(source, tmp_path / "second.tar.gz")

    assert first.read_bytes() == second.read_bytes()
    with tarfile.open(first, "r:gz") as archive:
        assert archive.getnames() == [
            "config.json",
            "metadata.json",
            "model.safetensors",
            "tokenizer.json",
        ]


def test_package_transformer_requires_model_weights(tmp_path: Path):
    source = tmp_path / "distilbert"
    _create_transformer_directory(source)
    source.joinpath("model.safetensors").unlink()

    with pytest.raises(ValueError, match="missing model.safetensors"):
        package_transformer(source, tmp_path / "model.tar.gz")
