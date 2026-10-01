"""Run a small, hand-labelled challenge set through a prediction adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from .metrics import evaluate_predictions


class Predictor(Protocol):
    def predict(self, text: str): ...


def evaluate_transformer_challenge(
    predictor: Predictor,
    cases: Sequence[Mapping[str, str]],
) -> dict:
    """Record per-case outputs, errors, and confidence calibration metrics."""

    expected = [case["label"] for case in cases]
    outputs = [predictor.predict(case["text"]) for case in cases]
    predictions = [output.label for output in outputs]
    confidence = [float(output.confidence) for output in outputs]
    model_name = str(outputs[0].model) if outputs else "unknown"
    model_version = str(outputs[0].model_version) if outputs else "unknown"
    metrics = evaluate_predictions(
        expected,
        predictions,
        confidence,
        model=model_name,
        model_version=model_version,
    )
    rows = [
        {
            "id": case["id"],
            "expected_label": expected_label,
            "predicted_label": prediction,
            "confidence": score,
            "correct": expected_label == prediction,
            "reference_url": case.get("reference_url", ""),
            "notes": case.get("notes", ""),
        }
        for case, expected_label, prediction, score in zip(
            cases, expected, predictions, confidence, strict=True
        )
    ]
    return {
        "evaluation_type": "small_hand_labelled_factuality_challenge",
        "warning": (
            "Diagnostic only: this small, hand-written set is not a held-out ISOT "
            "test set and is not representative of deployed news articles."
        ),
        "model": model_name,
        "model_version": model_version,
        "sample_count": len(rows),
        "accuracy": metrics.accuracy,
        "macro_f1": metrics.f1_macro,
        "expected_calibration_error": metrics.calibration[
            "expected_calibration_error"
        ],
        "mean_confidence": metrics.confidence_distribution["mean"],
        "errors": [row for row in rows if not row["correct"]],
        "predictions": rows,
    }


__all__ = ["evaluate_transformer_challenge"]
