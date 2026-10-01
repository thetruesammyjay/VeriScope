from types import SimpleNamespace

from ml.evaluation.transformer_challenge import evaluate_transformer_challenge


class StubPredictor:
    def __init__(self, predictions):
        self.predictions = iter(predictions)

    def predict(self, text):
        del text
        label, confidence = next(self.predictions)
        return SimpleNamespace(
            label=label,
            confidence=confidence,
            model="transformer_sequence_classifier",
            model_version="test-version",
        )


def test_challenge_evaluation_records_errors_and_confidence_without_cutoffs():
    report = evaluate_transformer_challenge(
        StubPredictor(
            [
                ("likely_fake", 0.99),
                ("likely_fake", 0.72),
            ]
        ),
        [
            {"id": "nasa-true", "text": "NASA sample", "label": "likely_real"},
            {"id": "false-claim", "text": "False claim", "label": "likely_fake"},
        ],
    )

    assert report["accuracy"] == 0.5
    assert report["errors"][0]["id"] == "nasa-true"
    assert report["errors"][0]["confidence"] == 0.99
    assert report["sample_count"] == 2
