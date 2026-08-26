from __future__ import annotations

import numpy as np
import pandas as pd

from ml.transformer.config import TransformerConfig
from ml.transformer.dataset import tokenize_frame
from ml.transformer.train import compute_metrics


class DummyTokenizer:
    def __call__(self, texts, *, truncation, max_length):
        assert truncation is True
        assert max_length == 16
        return {
            "input_ids": [[1, 2] for _ in texts],
            "attention_mask": [[1, 1] for _ in texts],
        }


def test_tokenize_frame_maps_public_labels_to_model_ids() -> None:
    frame = pd.DataFrame(
        {"text": ["fake story", "real report"], "label": ["fake", "real"]}
    )
    dataset = tokenize_frame(
        frame,
        DummyTokenizer(),
        config=TransformerConfig(max_length=16),
    )

    assert dataset.column_names == ["labels", "input_ids", "attention_mask"]
    assert dataset["labels"] == [0, 1]


def test_compute_metrics_returns_macro_scores() -> None:
    scores = compute_metrics((np.asarray([[2.0, 0.1], [0.1, 2.0]]), np.asarray([0, 1])))

    assert scores == {
        "accuracy": 1.0,
        "precision_macro": 1.0,
        "recall_macro": 1.0,
        "f1_macro": 1.0,
    }
