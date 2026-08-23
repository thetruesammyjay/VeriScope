"""Load validated model artifacts for API inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


def load_artifact(path: Path) -> dict[str, Any]:
    """Load one persisted artifact and validate its top-level shape."""

    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or "pipeline" not in artifact or "metadata" not in artifact:
        raise ValueError(f"Invalid model artifact: {path}")
    return artifact


__all__ = ["load_artifact"]
