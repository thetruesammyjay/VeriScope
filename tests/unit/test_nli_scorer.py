import pytest
from ml.verification.nli_scorer import normalize_nli_label, result_from_logits


def test_result_from_logits_maps_nli_relations() -> None:
    result = result_from_logits(
        [0.1, 0.2, 3.0],
        {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"},
    )

    assert result.relation == "entailment"
    assert set(result.probabilities) == {"contradiction", "neutral", "entailment"}
    assert sum(result.probabilities.values()) == pytest.approx(1.0)


def test_nli_scorer_rejects_unknown_label_names() -> None:
    with pytest.raises(ValueError, match="unsupported label"):
        normalize_nli_label("LABEL_4")
