"""Evaluate the selected model on a held-out test split."""

from __future__ import annotations

import argparse
from pathlib import Path

from .metrics import evaluate_csv, save_evaluation_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-csv", type=Path, default=Path("datasets/processed/test.csv")
    )
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=Path("models/classical/model.joblib"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/metrics/classical"),
    )
    args = parser.parse_args()

    result = evaluate_csv(args.test_csv, args.artifact_path)
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


__all__ = ["main"]
