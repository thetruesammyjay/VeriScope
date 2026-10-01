"""Evaluate an optional NLI checkpoint on the small claim/passage fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.verification.nli_scorer import NliScorer
from sklearn.metrics import accuracy_score, f1_score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Keep Hugging Face repo IDs as strings; pathlib.Path rewrites the slash on Windows.
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("tests/fixtures/nli_relation_eval.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/metrics/nli_relation_eval.json"),
    )
    parser.add_argument("--device")
    args = parser.parse_args()
    dataset = json.loads(args.cases.read_text(encoding="utf-8"))
    scorer = NliScorer(args.model_path, device=args.device)
    cases = dataset["cases"]
    outputs = scorer.score_many(
        [(case["passage"], case["claim"]) for case in cases]
    )
    predictions = []
    for case, output in zip(cases, outputs, strict=True):
        predictions.append(
            {
                **case,
                "predicted_relation": output.relation,
                "probabilities": output.probabilities,
                "correct": output.relation == case["relation"],
            }
        )
    expected = [case["relation"] for case in predictions]
    predicted = [case["predicted_relation"] for case in predictions]
    report = {
        "evaluation_type": "small_hand_labelled_nli_relation_diagnostic",
        "warning": (
            "Diagnostic only. NLI predicts text relations, not real-world truth. "
            "This fixture is too small for production claims."
        ),
        "accuracy": float(accuracy_score(expected, predicted)),
        "macro_f1": float(f1_score(expected, predicted, average="macro")),
        "sample_count": len(predictions),
        "predictions": predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved NLI diagnostic to {args.output}")


if __name__ == "__main__":
    main()
