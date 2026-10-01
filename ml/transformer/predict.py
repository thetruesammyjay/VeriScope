"""Inference for persisted transformer classifier artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .cleaning import clean_article_text


@dataclass(frozen=True)
class TransformerPrediction:
    label: str
    confidence: float
    model: str
    model_version: str
    processing_time_ms: float
    confidence_method: str = "raw_softmax"
    disclaimer: str = (
        "This is a machine-learning prediction and should not be treated as "
        "independent factual verification."
    )


class TransformerPredictor:
    """Load a saved transformer directory and predict one text input."""

    def __init__(self, artifact_path: Path, *, device: str | None = None) -> None:
        self.artifact_path = Path(artifact_path)
        self._tokenizer = AutoTokenizer.from_pretrained(self.artifact_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.artifact_path)
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._model.to(self._device).eval()
        metadata_path = self.artifact_path / "metadata.json"
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        )
        self.model = str(metadata.get("model", "transformer_sequence_classifier"))
        self.model_version = str(metadata.get("model_version", "unknown"))
        self._max_length = int(metadata.get("max_length", 256))
        self._strip_agency_markers = (
            metadata.get("text_preprocessing") == "strip_boundary_agency_markers_v1"
        )
        calibration = metadata.get("calibration", {})
        self._temperature = float(calibration.get("temperature", 1.0))
        self.confidence_method = (
            "temperature_scaled"
            if calibration.get("method") == "temperature_scaling"
            else "raw_softmax"
        )

    def predict(self, text: str) -> TransformerPrediction:
        started = perf_counter()
        model_text = clean_article_text(text) if self._strip_agency_markers else text
        encoded = self._tokenizer(
            model_text,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_length,
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = self._model(**encoded).logits[0] / self._temperature
            probabilities = torch.softmax(logits, dim=-1)
        index = int(torch.argmax(probabilities).item())
        raw_labels = self._model.config.id2label
        label = str(raw_labels.get(index, index)).lower()
        return TransformerPrediction(
            label=label,
            confidence=float(probabilities[index].item()),
            model=self.model,
            model_version=self.model_version,
            processing_time_ms=(perf_counter() - started) * 1_000,
            confidence_method=self.confidence_method,
        )


__all__ = ["TransformerPrediction", "TransformerPredictor"]
