import numpy as np
from ml.transformer.calibration import fit_temperature, negative_log_likelihood


def test_temperature_scaling_reduces_held_out_nll_for_overconfident_logits() -> None:
    logits = np.asarray([[8.0, 0.0], [0.0, 8.0], [5.0, 0.0], [0.0, 5.0]])
    labels = np.asarray([0, 1, 1, 0])

    temperature = fit_temperature(logits, labels)

    assert temperature > 1
    assert negative_log_likelihood(logits / temperature, labels) < negative_log_likelihood(
        logits, labels
    )
