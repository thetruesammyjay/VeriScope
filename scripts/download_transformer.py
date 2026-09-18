"""Download and extract the configured transformer release artifact."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from scripts.download_model import download_artifact, extract_tar_artifact


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.getenv("TRANSFORMER_MODEL_ARTIFACT_URL"))
    parser.add_argument(
        "--archive-path",
        type=Path,
        default=Path("models/transformer/distilbert.tar.gz"),
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=Path(os.getenv("TRANSFORMER_MODEL_PATH", "models/transformer/distilbert")),
    )
    parser.add_argument("--sha256", default=os.getenv("TRANSFORMER_MODEL_ARTIFACT_SHA256"))
    parser.add_argument("--token", default=os.getenv("TRANSFORMER_MODEL_ARTIFACT_TOKEN"))
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--required", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.url:
        if args.required:
            raise RuntimeError(
                "TRANSFORMER_MODEL_ARTIFACT_URL is required but is not configured"
            )
        print("TRANSFORMER_MODEL_ARTIFACT_URL is not set; skipping transformer download.")
        return 0

    archive_path = download_artifact(
        args.url,
        args.archive_path,
        sha256=args.sha256,
        token=args.token,
        timeout_seconds=args.timeout,
    )
    model_path = extract_tar_artifact(archive_path, args.extract_dir)
    print(f"Transformer model artifact extracted to {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
