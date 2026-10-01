"""Optional experimental NLI scoring for claim/passage pairs.

This scorer is deliberately not wired into the production verifier: NLI labels
describe a text-to-text relation and are not a determination of real-world truth.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

NLI_LABELS = ("contradiction", "neutral", "entailment")


@dataclass(frozen=True)
class NliResult:
    relation: str
    probabilities: dict[str, float]


def normalize_nli_label(value: str) -> str:
    normalized = value.lower().replace("label_", "").strip()
    aliases = {
        "contradiction": "contradiction",
        "contradictory": "contradiction",
        "neutral": "neutral",
        "entailment": "entailment",
        "entailed": "entailment",
    }
    if normalized not in aliases:
        raise ValueError(f"NLI model exposes an unsupported label: {value}")
    return aliases[normalized]


def result_from_logits(logits: list[float], id2label: dict[int, str]) -> NliResult:
    if len(logits) != len(id2label):
        raise ValueError("NLI logits and label mapping must have equal lengths")
    labels = {index: normalize_nli_label(id2label[index]) for index in range(len(logits))}
    if set(labels.values()) != set(NLI_LABELS):
        raise ValueError("NLI model must provide contradiction, neutral, and entailment labels")
    values = torch.softmax(torch.tensor(logits, dtype=torch.float64), dim=0).tolist()
    probabilities = {labels[index]: float(value) for index, value in enumerate(values)}
    relation = max(probabilities, key=probabilities.__getitem__)
    return NliResult(relation=relation, probabilities=probabilities)


class NliScorer:
    """Score premise passages against hypotheses using a local NLI checkpoint."""

    def __init__(self, model_path: str | Path, *, device: str | None = None) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._model.to(self._device).eval()
        self._id2label = {int(index): label for index, label in self._model.config.id2label.items()}
        for label in self._id2label.values():
            normalize_nli_label(label)

    def score(self, premise: str, hypothesis: str) -> NliResult:
        return self.score_many([(premise, hypothesis)])[0]

    def score_many(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        batch_size: int = 8,
    ) -> list[NliResult]:
        """Score several pairs in batches to avoid repeated CPU model startup."""

        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        results: list[NliResult] = []
        with torch.inference_mode():
            for start in range(0, len(pairs), batch_size):
                batch = pairs[start : start + batch_size]
                encoded = self._tokenizer(
                    [premise for premise, _ in batch],
                    [hypothesis for _, hypothesis in batch],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                encoded = {
                    key: value.to(self._device) for key, value in encoded.items()
                }
                logits = self._model(**encoded).logits.cpu().tolist()
                results.extend(
                    result_from_logits(row, self._id2label) for row in logits
                )
        return results


__all__ = ["NliResult", "NliScorer", "normalize_nli_label", "result_from_logits"]
