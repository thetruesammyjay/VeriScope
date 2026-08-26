"""Inference for persisted transformer classifier artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass(frozen=True)
class TransformerPrediction:
    label: str
    confidence: float
    model: str
    model_version: str
    processing_time_ms: float
    disclaimer: str = (
        "This is a machine-learning prediction and should not be treated as "
        "independent factual verification."
    )


class TransformerPredictor:
    """Load a saved transformer directory and predict one text input."""

    def __init__(self, artifact_path: Path, *, device: str | None = None) -> None:
        self.artifact_path = Path(artifact_path)
        self._tokenizer = AutoTokenizer.from_pretrained(self.artifact_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.artifact_path
        )
        self._device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._model.to(self._device).eval()
        metadata_path = self.artifact_path / "metadata.json"
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else {}
        )
        self.model = str(metadata.get("model", "transformer_sequence_classifier"))
        self.model_version = str(metadata.get("model_version", "unknown"))

    def predict(self, text: str) -> TransformerPrediction:
        started = perf_counter()
        encoded = self._tokenizer(text, return_tensors="pt", truncation=True)
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        with torch.inference_mode():
            probabilities = torch.softmax(self._model(**encoded).logits, dim=-1)[0]
        index = int(torch.argmax(probabilities).item())
        raw_labels = self._model.config.id2label
        label = str(raw_labels.get(index, index)).lower()
        return TransformerPrediction(
            label=label,
            confidence=float(probabilities[index].item()),
            model=self.model,
            model_version=self.model_version,
            processing_time_ms=(perf_counter() - started) * 1_000,
        )


__all__ = ["TransformerPrediction", "TransformerPredictor"]
