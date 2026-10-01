"""Evaluate DistilBERT on the small factuality diagnostic challenge set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.evaluation.transformer_challenge import evaluate_transformer_challenge
from ml.transformer.predict import TransformerPredictor

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "tests/fixtures/transformer_challenge_set.json"
DEFAULT_OUTPUT = ROOT / "reports/metrics/transformer_challenge/report.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=Path("models/transformer/distilbert"),
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()

    dataset = json.loads(args.cases.read_text(encoding="utf-8"))
    predictor = TransformerPredictor(args.artifact_path, device=args.device)
    report = evaluate_transformer_challenge(predictor, dataset["cases"])
    report["dataset_description"] = dataset["description"]
    report["label_basis"] = dataset["label_basis"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"{report['model']} ({report['model_version']}): "
        f"{report['sample_count']} samples; accuracy={report['accuracy']:.3f}; "
        f"ECE={report['expected_calibration_error']:.3f}"
    )
    print(f"Saved diagnostic report to {args.output}")


if __name__ == "__main__":
    main()
