"""Create a reproducible, release-ready transformer artifact archive."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import tarfile
import tempfile
from pathlib import Path

CHUNK_SIZE = 1024 * 1024
REQUIRED_FILES = {"config.json", "metadata.json", "tokenizer.json"}
MODEL_FILES = {"model.safetensors", "pytorch_model.bin"}


def sha256(path: Path) -> str:
    """Return the SHA-256 digest for a release asset."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_transformer(source: Path, output: Path) -> Path:
    """Archive a trained transformer directory with stable archive metadata."""

    source = Path(source)
    output = Path(output)
    if not source.is_dir():
        raise ValueError(f"Transformer directory does not exist: {source}")

    present = {path.name for path in source.iterdir() if path.is_file()}
    missing = REQUIRED_FILES - present
    if missing:
        raise ValueError(f"Transformer directory is missing required files: {', '.join(sorted(missing))}")
    if not present.intersection(MODEL_FILES):
        raise ValueError("Transformer directory is missing model.safetensors or pytorch_model.bin")

    files = sorted(path for path in source.rglob("*") if path.is_file() or path.is_symlink())
    if not files:
        raise ValueError("Transformer directory is empty")
    if any(path.is_symlink() for path in files):
        raise ValueError("Transformer directory must not contain symbolic links")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with temporary_path.open("wb") as destination:
            with gzip.GzipFile(
                filename="", fileobj=destination, mode="wb", mtime=0
            ) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                    for path in files:
                        info = archive.gettarinfo(
                            str(path), arcname=path.relative_to(source).as_posix()
                        )
                        info.uid = info.gid = info.mtime = 0
                        info.uname = info.gname = ""
                        with path.open("rb") as handle:
                            archive.addfile(info, handle)
        temporary_path.replace(output)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("models/transformer/distilbert")
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    artifact = package_transformer(args.source, args.output)
    print(f"Artifact: {artifact}")
    print(f"SHA256: {sha256(artifact)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "package_transformer", "sha256"]
