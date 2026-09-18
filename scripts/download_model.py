"""Download and verify a versioned model artifact for deployment builds.

The script intentionally uses Python's standard library so it can run before
the application is imported.  A missing ``MODEL_ARTIFACT_URL`` is treated as a
local-development no-op; Render should set that variable in its environment.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tarfile
import tempfile
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

CHUNK_SIZE = 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_checksum(path: Path, expected: str | None) -> None:
    if not expected:
        return
    normalized = expected.strip().lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError("MODEL_ARTIFACT_SHA256 must be a 64-character hexadecimal digest")
    actual = _sha256(path)
    if actual != normalized:
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected {normalized}, got {actual}"
        )


def download_artifact(
    url: str,
    destination: Path,
    *,
    sha256: str | None = None,
    token: str | None = None,
    timeout_seconds: float = 120.0,
) -> Path:
    """Download ``url`` to ``destination`` atomically and verify its digest."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.is_file() and (not sha256 or _sha256(destination) == sha256.strip().lower()):
        return destination

    headers = {"User-Agent": "automated-fake-news-detection-model-fetcher"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)

    temporary_path: Path | None = None
    try:
        with (
            urlopen(request, timeout=timeout_seconds) as response,
            tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary,
        ):
                temporary_path = Path(temporary.name)
                for chunk in iter(lambda: response.read(CHUNK_SIZE), b""):
                    temporary.write(chunk)
        _validate_checksum(temporary_path, sha256)
        temporary_path.replace(destination)
    except (HTTPError, URLError) as error:
        raise RuntimeError(f"Unable to download model artifact from {url}") from error
    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()

    return destination


def extract_tar_artifact(archive_path: Path, destination: Path) -> Path:
    """Safely and atomically extract a model ``.tar`` or ``.tar.gz`` artifact.

    Archives are untrusted deployment input, even when retrieved from a GitHub
    Release. Symlinks, hard links, devices, and paths outside the staging
    directory are rejected before any file is extracted.
    """

    archive_path = Path(archive_path)
    destination = Path(destination)
    if not tarfile.is_tarfile(archive_path):
        raise ValueError(f"Transformer artifact is not a valid tar archive: {archive_path}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.extract-", dir=destination.parent)
    )
    extracted = staging_root / "artifact"
    extracted.mkdir()
    resolved_root = extracted.resolve()
    backup: Path | None = None
    try:
        with tarfile.open(archive_path, mode="r:*") as archive:
            members = archive.getmembers()
            for member in members:
                member_path = (extracted / member.name).resolve()
                if not member_path.is_relative_to(resolved_root):
                    raise ValueError("Transformer archive contains an unsafe path")
                if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                    raise ValueError("Transformer archive contains an unsupported link or device")
            archive.extractall(extracted, members=members)

        entries = list(extracted.iterdir())
        source = entries[0] if len(entries) == 1 and entries[0].is_dir() else extracted
        if not any(source.iterdir()):
            raise ValueError("Transformer archive is empty")

        if destination.exists():
            backup = destination.with_name(f".{destination.name}.backup-{uuid.uuid4().hex}")
            destination.replace(backup)
        source.replace(destination)
        if backup:
            shutil.rmtree(backup)
        return destination
    except Exception:
        if backup and backup.exists() and not destination.exists():
            backup.replace(destination)
        raise
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.getenv("MODEL_ARTIFACT_URL"))
    parser.add_argument(
        "--destination",
        default=os.getenv("CLASSICAL_MODEL_PATH", "models/classical/model.joblib"),
    )
    parser.add_argument("--sha256", default=os.getenv("MODEL_ARTIFACT_SHA256"))
    parser.add_argument("--token", default=os.getenv("MODEL_ARTIFACT_TOKEN"))
    parser.add_argument(
        "--extract-dir",
        type=Path,
        help="Safely extract the downloaded tar artifact into this directory.",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--required",
        action="store_true",
        default=os.getenv("MODEL_ARTIFACT_REQUIRED", "false").lower()
        in {"1", "true", "yes"},
        help="Fail when MODEL_ARTIFACT_URL is not configured.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.url:
        if args.required:
            raise RuntimeError("MODEL_ARTIFACT_URL is required but is not configured")
        print("MODEL_ARTIFACT_URL is not set; skipping model download.")
        return 0

    path = download_artifact(
        args.url,
        Path(args.destination),
        sha256=args.sha256,
        token=args.token,
        timeout_seconds=args.timeout,
    )
    if args.extract_dir:
        path = extract_tar_artifact(path, args.extract_dir)
        print(f"Model artifact extracted to {path}")
    else:
        print(f"Model artifact ready at {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["download_artifact", "extract_tar_artifact", "main"]
