from ml.classical.predict import ClassicalPredictor
from ml.classical.train import train_model


def test_train_model_produces_probability_prediction():
    artifact = train_model(
        [
            "official report confirms the result",
            "verified statement from a public agency",
            "shocking secret cure hidden from doctors",
            "aliens control the election with magic machines",
        ],
        ["real", "real", "fake", "fake"],
    )

    prediction = ClassicalPredictor(artifact).predict(
        "official verified report from a public agency"
    )

    assert prediction.label in {"likely_real", "likely_fake"}
    assert 0 <= prediction.confidence <= 1
    assert prediction.model == "tfidf_logistic_regression"
