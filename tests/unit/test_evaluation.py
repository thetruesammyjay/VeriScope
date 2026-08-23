from pathlib import Path

import pandas as pd
import pytest

from ml.classical.train import save_artifact, train_model
from ml.evaluation.metrics import (
    evaluate_csv,
    evaluate_predictions,
    save_evaluation_report,
)


def test_evaluate_predictions_calculates_metrics_and_confusion_matrix():
    result = evaluate_predictions(
        ["likely_real", "likely_real", "likely_fake", "likely_fake"],
        ["likely_real", "likely_fake", "likely_fake", "likely_fake"],
        [0.9, 0.55, 0.8, 0.7],
        model="test-model",
        model_version="test-1",
    )

    assert result.sample_count == 4
    assert result.accuracy == pytest.approx(0.75)
    assert result.confusion_matrix == ((2, 0), (1, 1))
    assert result.model == "test-model"
    assert sum(bin_data["count"] for bin_data in result.confidence_distribution["bins"]) == 4
    assert "expected_calibration_error" in result.calibration


def test_evaluate_predictions_calculates_brier_score():
    result = evaluate_predictions(
        ["likely_real", "likely_fake"],
        ["likely_real", "likely_fake"],
        [0.9, 0.8],
        positive_probabilities=[0.9, 0.2],
    )

    assert result.calibration["brier_score"] == pytest.approx(0.025)


def test_evaluate_csv_and_save_report(tmp_path: Path):
    artifact = train_model(
        [
            "official agency confirms the public report",
            "verified ministry announcement was published",
            "secret miracle cure hidden from doctors",
            "aliens control the election with magic machines",
        ],
        ["likely_real", "likely_real", "likely_fake", "likely_fake"],
    )
    artifact_path = save_artifact(artifact, tmp_path / "model.joblib")
    test_csv = tmp_path / "test.csv"
    pd.DataFrame(
        {
            "text": [
                "official agency confirms the public report",
                "secret miracle cure hidden from doctors",
            ],
            "label": ["likely_real", "likely_fake"],
        }
    ).to_csv(test_csv, index=False)

    result = evaluate_csv(test_csv, artifact_path)
    paths = save_evaluation_report(result, tmp_path / "report")

    assert result.sample_count == 2
    assert 0 <= result.accuracy <= 1
    assert all(path.is_file() for path in paths.values())
    assert (tmp_path / "report" / "metrics.json").read_text(encoding="utf-8")
