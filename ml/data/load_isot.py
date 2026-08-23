"""Load the original ISOT fake-news CSV files.

The ISOT dataset is distributed as two files, ``Fake.csv`` and ``True.csv``.
This module keeps ingestion separate from cleaning and splitting so each step
can be tested and reproduced independently.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ISOT_FILES = {"likely_fake": "Fake.csv", "likely_real": "True.csv"}
OPTIONAL_COLUMNS = ("title", "subject", "date")


def _find_file(raw_dir: Path, filename: str) -> Path:
    """Find a dataset file case-insensitively on case-sensitive systems."""

    direct = raw_dir / filename
    if direct.is_file():
        return direct
    matches = [
        path
        for path in raw_dir.iterdir()
        if path.is_file() and path.name.casefold() == filename.casefold()
    ]
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Expected ISOT file {filename!r} in {raw_dir}")


def _read_csv(path: Path) -> pd.DataFrame:
    """Read an ISOT CSV, falling back for legacy non-UTF-8 copies."""

    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def load_isot_file(path: Path, label: str) -> pd.DataFrame:
    """Load one ISOT CSV and attach its canonical label."""

    frame = _read_csv(Path(path))
    required = "text"
    if required not in frame.columns:
        raise ValueError(f"ISOT file {path} must contain a 'text' column")
    for column in OPTIONAL_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    frame["label"] = label
    frame["source_file"] = Path(path).name
    return frame[["title", "text", "subject", "date", "label", "source_file"]]


def load_isot_dataset(raw_dir: Path = Path("datasets/raw/isot")) -> pd.DataFrame:
    """Load and concatenate the fake and real ISOT files."""

    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"ISOT directory does not exist: {raw_dir}")
    frames = [
        load_isot_file(_find_file(raw_dir, filename), label)
        for label, filename in ISOT_FILES.items()
    ]
    return pd.concat(frames, ignore_index=True)


# Short alias for callers that prefer the dataset name over the format name.
load_isot = load_isot_dataset


__all__ = ["ISOT_FILES", "load_isot", "load_isot_dataset", "load_isot_file"]
