"""Fit temperature on calibration data and assess once on held-out test data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ml.data.validate import canonical_label
from ml.transformer.calibration import fit_temperature, negative_log_likelihood
from ml.transformer.cleaning import clean_article_text
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _load_logits(model, tokenizer, texts, *, device: torch.device) -> np.ndarray:
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(texts), 16):
            encoded = tokenizer(
                texts[start : start + 16],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            chunks.append(model(**encoded).logits.cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _report(
    logits: np.ndarray,
    labels: np.ndarray,
    temperature: float,
    confidence_cutpoints: list[tuple[float, float]],
) -> dict:
    probabilities_raw = torch.softmax(torch.as_tensor(logits), dim=-1).numpy()
    probabilities_scaled = torch.softmax(torch.as_tensor(logits / temperature), dim=-1).numpy()
    rows = {}
    for name, probabilities in (
        ("uncalibrated", probabilities_raw),
        ("temperature_scaled", probabilities_scaled),
    ):
        predictions = probabilities.argmax(axis=1)
        confidence = probabilities.max(axis=1)
        correctness = predictions == labels
        bin_edges = np.linspace(0.0, 1.0, 11)
        ece = 0.0
        for lower, upper in zip(bin_edges[:-1], bin_edges[1:], strict=True):
            mask = (confidence >= lower) & (
                confidence <= upper if upper == 1.0 else confidence < upper
            )
            if mask.any():
                ece += float(mask.mean()) * abs(
                    float(correctness[mask].mean()) - float(confidence[mask].mean())
                )
        rows[name] = {
            "accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, average="macro")),
            "nll": negative_log_likelihood(
                logits if name == "uncalibrated" else logits / temperature, labels
            ),
            "mean_max_probability": float(confidence.mean()),
            "expected_calibration_error": ece,
            "multiclass_brier_score": float(
                np.mean(
                    np.sum(
                        (probabilities - np.eye(probabilities.shape[1], dtype=float)[labels]) ** 2,
                        axis=1,
                    )
                )
            ),
            "risk_coverage_at_calibration_quantiles": [
                {
                    "calibration_quantile": quantile,
                    "minimum_confidence": threshold,
                    "coverage": float((confidence >= threshold).mean()),
                    "selective_error_rate": (
                        float(1 - correctness[confidence >= threshold].mean())
                        if (confidence >= threshold).any()
                        else None
                    ),
                }
                for quantile, threshold in confidence_cutpoints
            ],
        }
    return rows


def calibrate(
    artifact_path: Path,
    calibration_csv: Path,
    test_csv: Path,
    *,
    device_name: str | None = None,
) -> dict:
    artifact_path = Path(artifact_path)
    tokenizer = AutoTokenizer.from_pretrained(artifact_path)
    model = AutoModelForSequenceClassification.from_pretrained(artifact_path)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).eval()
    metadata_path = artifact_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    label_ids = {
        canonical_label(label): int(index) for index, label in model.config.id2label.items()
    }

    def read(path: Path):
        frame = pd.read_csv(path)
        texts = frame["text"].astype(str).tolist()
        if metadata.get("text_preprocessing") == "strip_boundary_agency_markers_v1":
            texts = [clean_article_text(text) for text in texts]
        labels = np.asarray(
            [label_ids[canonical_label(label)] for label in frame["label"]],
            dtype=np.int64,
        )
        return _load_logits(model, tokenizer, texts, device=device), labels

    calibration_logits, calibration_labels = read(calibration_csv)
    test_logits, test_labels = read(test_csv)
    temperature = fit_temperature(calibration_logits, calibration_labels)
    calibration_probabilities = torch.softmax(
        torch.as_tensor(calibration_logits / temperature), dim=-1
    ).numpy()
    calibration_confidence = calibration_probabilities.max(axis=1)
    confidence_cutpoints = [
        (quantile, float(np.quantile(calibration_confidence, quantile)))
        for quantile in (0.5, 0.75, 0.9, 0.95)
    ]
    before = negative_log_likelihood(calibration_logits, calibration_labels)
    after = negative_log_likelihood(calibration_logits / temperature, calibration_labels)
    report = {
        "evaluation_type": "held_out_temperature_scaling",
        "warning": (
            "Calibration is fitted only on the calibration split and assessed on a "
            "separate test split. It measures in-distribution ISOT scores, not "
            "factual truth or cross-domain calibration."
        ),
        "artifact": str(artifact_path),
        "calibration_count": int(len(calibration_labels)),
        "test_count": int(len(test_labels)),
        "temperature": temperature,
        "calibration_nll_before": before,
        "calibration_nll_after": after,
        "calibration_metrics": _report(
            calibration_logits,
            calibration_labels,
            temperature,
            confidence_cutpoints,
        ),
        "test_metrics": _report(
            test_logits,
            test_labels,
            temperature,
            confidence_cutpoints,
        ),
        "abstention_policy": {
            "threshold_selected": False,
            "reason": (
                "Risk/coverage is reported at validation-derived quantiles; a "
                "deployed abstention rule requires an explicit acceptable-error policy."
            ),
        },
    }
    metadata["calibration"] = {
        "method": "temperature_scaling",
        "temperature": temperature,
        "fitted_on": "calibration split; not the final test split",
        "dataset": "ISOT cleaned split",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", type=Path, required=True)
    parser.add_argument(
        "--calibration-csv",
        type=Path,
        default=Path("datasets/processed/transformer_source_clean/calibration.csv"),
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=Path("datasets/processed/transformer_source_clean/test.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/metrics/transformer_source_clean/calibration.json"),
    )
    parser.add_argument("--device")
    args = parser.parse_args()
    report = calibrate(
        args.artifact_path,
        args.calibration_csv,
        args.test_csv,
        device_name=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved held-out calibration report to {args.output}")


if __name__ == "__main__":
    main()
