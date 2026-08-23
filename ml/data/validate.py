"""Validation and reporting for processed news datasets."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = ("text", "label")
LABEL_ALIASES = {
    "fake": "likely_fake",
    "false": "likely_fake",
    "likely_fake": "likely_fake",
    "0": "likely_fake",
    "real": "likely_real",
    "true": "likely_real",
    "likely_real": "likely_real",
    "1": "likely_real",
}


class DatasetValidationError(ValueError):
    """Raised when a dataset cannot safely be used for training."""


@dataclass(frozen=True)
class DatasetReport:
    rows: int
    labels: dict[str, int]
    duplicate_rows: int


def canonical_label(value: Any) -> str:
    """Return the public label name for a supported dataset label."""

    normalized = str(value).strip().lower()
    try:
        return LABEL_ALIASES[normalized]
    except KeyError as error:
        raise DatasetValidationError(f"Unsupported label: {value!r}") from error


def _normalized_text(value: Any) -> str:
    return " ".join(re.sub(r"\s+", " ", str(value).strip().lower()).split())


def dataset_report(frame: pd.DataFrame) -> DatasetReport:
    """Produce non-raising summary information for a dataset."""

    if "text" not in frame or "label" not in frame:
        return DatasetReport(rows=len(frame), labels={}, duplicate_rows=0)
    keys = frame["text"].map(_normalized_text)
    return DatasetReport(
        rows=len(frame),
        labels=dict(Counter(canonical_label(value) for value in frame["label"])),
        duplicate_rows=int(keys.duplicated(keep=False).sum()),
    )


def validate_dataset(
    frame: pd.DataFrame,
    *,
    require_both_labels: bool = True,
    reject_duplicates: bool = True,
    min_text_length: int = 1,
) -> DatasetReport:
    """Validate schema, labels, usable text, and leakage-prone duplicates."""

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DatasetValidationError(
            f"Dataset is missing required columns: {', '.join(missing)}"
        )
    if frame.empty:
        raise DatasetValidationError("Dataset must contain at least one row")
    if min_text_length < 1:
        raise ValueError("min_text_length must be at least 1")

    labels = [canonical_label(value) for value in frame["label"]]
    text = frame["text"].map(_normalized_text)
    if (text.str.len() < min_text_length).any():
        raise DatasetValidationError("Dataset contains empty or too-short text")
    duplicate_rows = int(text.duplicated(keep=False).sum())
    if reject_duplicates and duplicate_rows:
        raise DatasetValidationError(
            f"Dataset contains {duplicate_rows} duplicate text rows"
        )
    if require_both_labels and len(set(labels)) < 2:
        raise DatasetValidationError(
            "Dataset must contain both likely_real and likely_fake labels"
        )
    return DatasetReport(
        rows=len(frame),
        labels=dict(Counter(labels)),
        duplicate_rows=duplicate_rows,
    )


validate_frame = validate_dataset


__all__ = [
    "DatasetReport",
    "DatasetValidationError",
    "canonical_label",
    "dataset_report",
    "validate_dataset",
    "validate_frame",
]
