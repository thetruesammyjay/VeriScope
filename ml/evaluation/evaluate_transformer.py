"""Evaluate a saved Hugging Face transformer on a labelled CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ml.data.validate import canonical_label, validate_dataset

from .metrics import EvaluationResult, evaluate_predictions, save_evaluation_report


def evaluate_transformer_csv(
    test_csv: Path,
    artifact_path: Path,
    *,
    batch_size: int = 16,
    max_length: int | None = None,
    device: str | None = None,
    text_column: str = "text",
    label_column: str = "label",
) -> EvaluationResult:
    """Run batched transformer inference and calculate common metrics."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    frame = pd.read_csv(test_csv)
    if text_column not in frame or label_column not in frame:
        raise ValueError(
            f"CSV must contain {text_column!r} and {label_column!r} columns"
        )
    validate_dataset(frame)

    artifact_path = Path(artifact_path)
    tokenizer = AutoTokenizer.from_pretrained(artifact_path)
    model = AutoModelForSequenceClassification.from_pretrained(artifact_path)
    resolved_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(resolved_device).eval()
    id_to_label = {
        int(index): canonical_label(label)
        for index, label in model.config.id2label.items()
    }
    metadata_path = artifact_path / "metadata.json"
    metadata: dict[str, Any] = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )

    predicted: list[str] = []
    confidence: list[float] = []
    positive_probabilities: list[float] = []
    texts = frame[text_column].astype(str).tolist()
    resolved_max_length = max_length or int(metadata.get("max_length", 256))
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start : start + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=resolved_max_length,
            )
            encoded = {
                key: value.to(resolved_device) for key, value in encoded.items()
            }
            probabilities = torch.softmax(model(**encoded).logits, dim=-1)
            values, indices = torch.max(probabilities, dim=-1)
            predicted.extend(id_to_label[int(index)] for index in indices.cpu().tolist())
            confidence.extend(float(value) for value in values.cpu().tolist())
            real_index = next(
                index for index, label in id_to_label.items() if label == "likely_real"
            )
            positive_probabilities.extend(
                float(value) for value in probabilities[:, real_index].cpu().tolist()
            )

    return evaluate_predictions(
        frame[label_column].tolist(),
        predicted,
        confidence,
        model=str(metadata.get("model", "transformer_sequence_classifier")),
        model_version=str(metadata.get("model_version", "unknown")),
        positive_probabilities=positive_probabilities,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-csv", type=Path, default=Path("datasets/processed/test.csv")
    )
    parser.add_argument(
        "--artifact-path", type=Path, required=True
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/metrics/transformer")
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()

    result = evaluate_transformer_csv(
        args.test_csv,
        args.artifact_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )
    paths = save_evaluation_report(result, args.output_dir)
    print(f"Model: {result.model} ({result.model_version})")
    print(f"Samples: {result.sample_count}")
    print(f"Accuracy: {result.accuracy:.4f}")
    print(f"Precision (macro): {result.precision_macro:.4f}")
    print(f"Recall (macro): {result.recall_macro:.4f}")
    print(f"F1-score (macro): {result.f1_macro:.4f}")
    print(f"Reports saved to {args.output_dir}")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()


__all__ = ["evaluate_transformer_csv", "main"]
