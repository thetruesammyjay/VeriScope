"""Dataset preparation for Hugging Face sequence-classification models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from datasets import Dataset
from ml.data.validate import canonical_label, validate_dataset

from .config import TransformerConfig


def load_frame(
    path: Path, *, text_column: str = "text", label_column: str = "label"
) -> pd.DataFrame:
    """Load and validate a CSV for transformer training."""

    frame = pd.read_csv(path)
    if text_column not in frame or label_column not in frame:
        raise ValueError(
            f"CSV must contain {text_column!r} and {label_column!r} columns"
        )
    prepared = (
        frame[[text_column, label_column]]
        .rename(columns={text_column: "text", label_column: "label"})
        .copy()
    )
    prepared["text"] = prepared["text"].astype(str)
    prepared["label"] = prepared["label"].map(canonical_label)
    validate_dataset(prepared)
    return prepared


def tokenize_frame(
    frame: pd.DataFrame,
    tokenizer: Any,
    *,
    config: TransformerConfig | None = None,
) -> Dataset:
    """Convert validated text/label rows into a tokenized HF Dataset."""

    resolved = config or TransformerConfig()
    prepared = frame[["text", "label"]].copy()
    prepared["text"] = prepared["text"].astype(str)
    prepared["label"] = prepared["label"].map(canonical_label).map(resolved.label2id)
    validate_dataset(
        prepared.assign(label=prepared["label"].map(resolved.id2label)),
    )
    dataset = Dataset.from_pandas(
        prepared.rename(columns={"label": "labels"}), preserve_index=False
    )

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=resolved.max_length,
        )

    return dataset.map(tokenize, batched=True, remove_columns=["text"])


__all__ = ["load_frame", "tokenize_frame"]
