"""Build a clean, deduplicated dataset from the original ISOT files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .load_isot import load_isot_dataset
from .validate import canonical_label, validate_dataset


def clean_text(value: Any) -> str:
    """Normalize whitespace and missing values without removing word signals."""

    if pd.isna(value):
        return ""
    return " ".join(re.sub(r"\s+", " ", str(value)).strip().split())


def _combine_title_and_body(title: Any, body: Any) -> str:
    cleaned_title = clean_text(title)
    cleaned_body = clean_text(body)
    if cleaned_title and cleaned_body:
        return f"{cleaned_title}. {cleaned_body}"
    return cleaned_title or cleaned_body


def deduplicate_records(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove exact normalized-text duplicates and ambiguous label conflicts.

    If the same text appears with both labels, all copies are removed rather
    than assigning a label based on file order. This prevents contradictory
    records from leaking into train and test splits.
    """

    working = frame.copy()
    keys = working["text"].map(
        lambda value: " ".join(str(value).strip().lower().split())
    )
    working["_dedupe_key"] = keys
    conflicting_keys = (
        working.groupby("_dedupe_key")["label"].nunique()
        .loc[lambda values: values > 1]
        .index
    )
    working = working[~working["_dedupe_key"].isin(conflicting_keys)]
    working = working.drop_duplicates(subset=["_dedupe_key"], keep="first")
    return working.drop(columns=["_dedupe_key"]).reset_index(drop=True)


def prepare_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Create the canonical text/label schema used by model training."""

    working = frame.copy()
    for column in ("title", "text", "subject", "date"):
        if column not in working:
            working[column] = ""
    working["title"] = working["title"].map(clean_text)
    working["body"] = working["text"].map(clean_text)
    working["text"] = [
        _combine_title_and_body(title, body)
        for title, body in zip(working["title"], working["body"])
    ]
    working["subject"] = working["subject"].map(clean_text)
    working["date"] = working["date"].map(clean_text)
    working["label"] = working["label"].map(canonical_label)
    working = working[working["text"].str.len() > 0]
    working = deduplicate_records(working)
    columns = ["text", "label", "title", "body", "subject", "date"]
    if "source_file" in working:
        columns.append("source_file")
    return working[columns]


def build_dataset(
    raw_dir: Path = Path("datasets/raw/isot"),
    output_path: Path = Path("datasets/processed/dataset.csv"),
) -> Path:
    """Load, clean, deduplicate, validate, and write the processed dataset."""

    prepared = prepare_dataset(load_isot_dataset(raw_dir))
    validate_dataset(prepared)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir", type=Path, default=Path("datasets/raw/isot")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("datasets/processed/dataset.csv")
    )
    args = parser.parse_args()
    path = build_dataset(args.raw_dir, args.output)
    print(f"Saved processed dataset to {path}")


if __name__ == "__main__":
    main()


__all__ = [
    "build_dataset",
    "clean_text",
    "deduplicate_records",
    "main",
    "prepare_dataset",
]
