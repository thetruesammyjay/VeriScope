"""Train and persist the TF-IDF + Logistic Regression baseline."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import ClassicalConfig
from .features import build_vectorizer, normalize_text


def canonical_label(label: object) -> str:
    """Map dataset labels to the public prediction labels."""

    value = str(label).strip().lower()
    if value in {"fake", "false", "likely_fake", "0"}:
        return "likely_fake"
    if value in {"real", "true", "likely_real", "1"}:
        return "likely_real"
    raise ValueError(f"Unsupported news label: {label!r}")


def train_model(
    texts: Iterable[str],
    labels: Iterable[object],
    *,
    config: ClassicalConfig | None = None,
) -> dict[str, object]:
    """Fit a classifier and return a serialisable model artifact."""

    resolved = config or ClassicalConfig()
    normalized_texts = [normalize_text(text) for text in texts]
    normalized_labels = [canonical_label(label) for label in labels]
    if len(normalized_texts) != len(normalized_labels):
        raise ValueError("texts and labels must contain the same number of items")
    if len(set(normalized_labels)) < 2:
        raise ValueError("training requires both likely_real and likely_fake labels")

    pipeline = Pipeline(
        steps=[
            ("tfidf", build_vectorizer(resolved)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=resolved.max_iter,
                    random_state=resolved.random_state,
                ),
            ),
        ]
    )
    pipeline.fit(normalized_texts, normalized_labels)
    return {
        "pipeline": pipeline,
        "metadata": {
            "model": "tfidf_logistic_regression",
            "model_version": resolved.model_version,
            "label_mapping": {
                "likely_real": "likely_real",
                "likely_fake": "likely_fake",
            },
            "feature_config": {
                "max_features": resolved.max_features,
                "ngram_range": resolved.ngram_range,
                "min_df": resolved.min_df,
                "max_df": resolved.max_df,
            },
        },
    }


def save_artifact(artifact: dict[str, object], path: Path) -> Path:
    """Persist a trained artifact and return its resolved path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    return path


def train_csv(
    csv_path: Path,
    *,
    text_column: str = "text",
    label_column: str = "label",
    artifact_path: Path | None = None,
    config: ClassicalConfig | None = None,
) -> Path:
    """Train from a processed CSV containing text and label columns."""

    frame = pd.read_csv(csv_path)
    if text_column not in frame or label_column not in frame:
        raise ValueError(
            f"CSV must contain {text_column!r} and {label_column!r} columns"
        )
    resolved = config or ClassicalConfig()
    artifact = train_model(frame[text_column].astype(str), frame[label_column], config=resolved)
    return save_artifact(artifact, artifact_path or resolved.artifact_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--artifact-path", type=Path, default=ClassicalConfig().artifact_path)
    args = parser.parse_args()
    path = train_csv(
        args.csv_path,
        text_column=args.text_column,
        label_column=args.label_column,
        artifact_path=args.artifact_path,
    )
    print(f"Saved classical model artifact to {path}")


if __name__ == "__main__":
    main()


__all__ = ["canonical_label", "main", "save_artifact", "train_csv", "train_model"]
