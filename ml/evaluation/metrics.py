"""Evaluation utilities for trained news-classification artifacts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.metrics import (
    confusion_matrix as sklearn_confusion_matrix,
)

from ml.classical.features import normalize_text
from ml.classical.train import canonical_label
from ml.data.validate import validate_dataset
from ml.inference.loader import load_artifact

CLASS_LABELS = ("likely_fake", "likely_real")
CONFIDENCE_EDGES = tuple(round(float(value), 2) for value in np.linspace(0.0, 1.0, 11))


@dataclass(frozen=True)
class EvaluationResult:
    """Serializable metrics and per-row outputs for one evaluation run."""

    model: str
    model_version: str
    sample_count: int
    labels: tuple[str, ...]
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    confusion_matrix: tuple[tuple[int, ...], ...]
    true_labels: tuple[str, ...]
    predicted_labels: tuple[str, ...]
    confidence_scores: tuple[float, ...]
    positive_probabilities: tuple[float, ...] | None = None

    @property
    def confidence_distribution(self) -> dict[str, Any]:
        """Return summary statistics and fixed-width confidence bins."""

        values = np.asarray(self.confidence_scores, dtype=float)
        counts, _ = np.histogram(values, bins=CONFIDENCE_EDGES)
        bins = [
            {
                "lower": CONFIDENCE_EDGES[index],
                "upper": CONFIDENCE_EDGES[index + 1],
                "count": int(count),
            }
            for index, count in enumerate(counts)
        ]
        return {
            "count": int(values.size),
            "min": float(values.min()) if values.size else 0.0,
            "max": float(values.max()) if values.size else 0.0,
            "mean": float(values.mean()) if values.size else 0.0,
            "median": float(np.median(values)) if values.size else 0.0,
            "std": float(values.std()) if values.size else 0.0,
            "bins": bins,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the report without repeating every row prediction."""

        return {
            "model": self.model,
            "model_version": self.model_version,
            "sample_count": self.sample_count,
            "labels": list(self.labels),
            "metrics": {
                "accuracy": self.accuracy,
                "precision_macro": self.precision_macro,
                "recall_macro": self.recall_macro,
                "f1_macro": self.f1_macro,
                "precision_weighted": self.precision_weighted,
                "recall_weighted": self.recall_weighted,
                "f1_weighted": self.f1_weighted,
            },
            "confusion_matrix": [list(row) for row in self.confusion_matrix],
            "calibration": self.calibration,
            "confidence_distribution": self.confidence_distribution,
        }

    @property
    def calibration(self) -> dict[str, Any]:
        """Return confidence calibration and optional Brier score metrics."""

        confidence = np.asarray(self.confidence_scores, dtype=float)
        correct = np.asarray(
            [predicted == actual for predicted, actual in zip(self.predicted_labels, self.true_labels)],
            dtype=float,
        )
        counts, _ = np.histogram(confidence, bins=CONFIDENCE_EDGES)
        ece = 0.0
        bins: list[dict[str, float | int]] = []
        for index, count in enumerate(counts):
            mask = (confidence >= CONFIDENCE_EDGES[index]) & (
                confidence <= CONFIDENCE_EDGES[index + 1]
                if index == len(counts) - 1
                else confidence < CONFIDENCE_EDGES[index + 1]
            )
            if not mask.any():
                continue
            bin_accuracy = float(correct[mask].mean())
            bin_confidence = float(confidence[mask].mean())
            weight = float(mask.mean())
            ece += weight * abs(bin_accuracy - bin_confidence)
            bins.append(
                {
                    "lower": CONFIDENCE_EDGES[index],
                    "upper": CONFIDENCE_EDGES[index + 1],
                    "count": int(mask.sum()),
                    "accuracy": bin_accuracy,
                    "mean_confidence": bin_confidence,
                }
            )
        result: dict[str, Any] = {"expected_calibration_error": ece, "bins": bins}
        if self.positive_probabilities is not None:
            positive = np.asarray(self.positive_probabilities, dtype=float)
            target = np.asarray(
                [label == "likely_real" for label in self.true_labels], dtype=float
            )
            result["brier_score"] = float(np.mean((positive - target) ** 2))
        return result


def evaluate_predictions(
    true_labels: Sequence[object],
    predicted_labels: Sequence[object],
    confidence_scores: Sequence[float],
    *,
    model: str = "unknown",
    model_version: str = "unknown",
    labels: Sequence[str] = CLASS_LABELS,
    positive_probabilities: Sequence[float] | None = None,
) -> EvaluationResult:
    """Calculate classification metrics from labels and confidence scores."""

    if not (
        len(true_labels) == len(predicted_labels) == len(confidence_scores)
    ):
        raise ValueError("labels and confidence_scores must have equal lengths")
    if not true_labels:
        raise ValueError("evaluation requires at least one prediction")

    normalized_true = tuple(canonical_label(value) for value in true_labels)
    normalized_predicted = tuple(canonical_label(value) for value in predicted_labels)
    confidence = np.asarray(confidence_scores, dtype=float)
    if np.any(~np.isfinite(confidence)) or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("confidence scores must be finite values between 0 and 1")
    positive_probability_values: tuple[float, ...] | None = None
    if positive_probabilities is not None:
        if len(positive_probabilities) != len(normalized_true):
            raise ValueError("positive_probabilities must match the label length")
        positive = np.asarray(positive_probabilities, dtype=float)
        if np.any(~np.isfinite(positive)) or np.any((positive < 0) | (positive > 1)):
            raise ValueError("positive probabilities must be finite values between 0 and 1")
        positive_probability_values = tuple(float(value) for value in positive)

    label_names = tuple(labels)
    matrix = sklearn_confusion_matrix(
        normalized_true,
        normalized_predicted,
        labels=label_names,
    )
    precision, recall, f1, _ = precision_recall_fscore_support(
        normalized_true,
        normalized_predicted,
        labels=label_names,
        average=None,
        zero_division=0,
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        normalized_true,
        normalized_predicted,
        labels=label_names,
        average="macro",
        zero_division=0,
    )
    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            normalized_true,
            normalized_predicted,
            labels=label_names,
            average="weighted",
            zero_division=0,
        )
    )
    del precision, recall, f1
    return EvaluationResult(
        model=model,
        model_version=model_version,
        sample_count=len(normalized_true),
        labels=label_names,
        accuracy=float(accuracy_score(normalized_true, normalized_predicted)),
        precision_macro=float(precision_macro),
        recall_macro=float(recall_macro),
        f1_macro=float(f1_macro),
        precision_weighted=float(precision_weighted),
        recall_weighted=float(recall_weighted),
        f1_weighted=float(f1_weighted),
        confusion_matrix=tuple(tuple(int(value) for value in row) for row in matrix),
        true_labels=normalized_true,
        predicted_labels=normalized_predicted,
        confidence_scores=tuple(float(value) for value in confidence),
        positive_probabilities=positive_probability_values,
    )


def evaluate_csv(
    test_csv: Path,
    artifact_path: Path,
    *,
    text_column: str = "text",
    label_column: str = "label",
) -> EvaluationResult:
    """Evaluate a persisted artifact against a labelled test CSV."""

    frame = pd.read_csv(test_csv)
    if text_column not in frame or label_column not in frame:
        raise ValueError(
            f"CSV must contain {text_column!r} and {label_column!r} columns"
        )
    validate_dataset(frame)
    artifact = load_artifact(Path(artifact_path))
    pipeline = artifact["pipeline"]
    if not hasattr(pipeline, "predict_proba"):
        raise ValueError("Model artifact pipeline must provide predict_proba")

    texts = [normalize_text(value) for value in frame[text_column].astype(str)]
    predicted = pipeline.predict(texts)
    probabilities = np.asarray(pipeline.predict_proba(texts), dtype=float)
    confidence = probabilities.max(axis=1)
    classes = [canonical_label(value) for value in pipeline.classes_]
    try:
        real_index = classes.index("likely_real")
    except ValueError as error:
        raise ValueError("Model pipeline must expose a likely_real class") from error
    metadata = artifact.get("metadata", {})
    return evaluate_predictions(
        frame[label_column].tolist(),
        predicted.tolist(),
        confidence.tolist(),
        model=str(metadata.get("model", "unknown")),
        model_version=str(metadata.get("model_version", "unknown")),
        positive_probabilities=probabilities[:, real_index].tolist(),
    )


def save_evaluation_report(result: EvaluationResult, output_dir: Path) -> dict[str, Path]:
    """Write JSON, CSV, and PNG evaluation artifacts."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(result.to_dict(), indent=2), encoding="utf-8"
    )

    confusion_path = output_dir / "confusion_matrix.csv"
    pd.DataFrame(result.confusion_matrix, index=result.labels, columns=result.labels).to_csv(
        confusion_path
    )
    distribution = result.confidence_distribution["bins"]
    distribution_path = output_dir / "confidence_distribution.csv"
    pd.DataFrame(distribution).to_csv(distribution_path, index=False)

    predictions_path = output_dir / "predictions.csv"
    pd.DataFrame(
        {
            "true_label": result.true_labels,
            "predicted_label": result.predicted_labels,
            "confidence": result.confidence_scores,
        }
    ).to_csv(predictions_path, index=False)

    confusion_plot_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(result, confusion_plot_path)
    confidence_plot_path = output_dir / "confidence_distribution.png"
    plot_confidence_distribution(result, confidence_plot_path)
    return {
        "metrics": metrics_path,
        "confusion_matrix": confusion_path,
        "confidence_distribution": distribution_path,
        "predictions": predictions_path,
        "confusion_matrix_plot": confusion_plot_path,
        "confidence_distribution_plot": confidence_plot_path,
    }


def plot_confusion_matrix(result: EvaluationResult, output_path: Path) -> None:
    """Save a labelled confusion-matrix heatmap."""

    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(result.confusion_matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=range(len(result.labels)),
        yticks=range(len(result.labels)),
        xticklabels=result.labels,
        yticklabels=result.labels,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion matrix",
    )
    for row_index, row in enumerate(result.confusion_matrix):
        for column_index, value in enumerate(row):
            axis.text(column_index, row_index, value, ha="center", va="center")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_confidence_distribution(result: EvaluationResult, output_path: Path) -> None:
    """Save a histogram of maximum predicted probabilities."""

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.hist(result.confidence_scores, bins=CONFIDENCE_EDGES, edgecolor="black")
    axis.set(
        xlim=(0, 1),
        xlabel="Prediction confidence",
        ylabel="Number of samples",
        title="Prediction confidence distribution",
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


__all__ = [
    "CLASS_LABELS",
    "EvaluationResult",
    "evaluate_csv",
    "evaluate_predictions",
    "plot_confidence_distribution",
    "plot_confusion_matrix",
    "save_evaluation_report",
]
