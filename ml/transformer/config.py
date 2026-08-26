"""Configuration for the transformer text-classification pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TransformerConfig:
    """Reproducible defaults for fine-tuning a sequence classifier."""

    model_name: str = "distilbert-base-uncased"
    max_length: int = 256
    train_batch_size: int = 8
    eval_batch_size: int = 16
    learning_rate: float = 2e-5
    num_train_epochs: float = 3.0
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    logging_steps: int = 50
    random_state: int = 42
    save_checkpoints: bool = False
    model_version: str = "transformer-distilbert-0.1.0"
    artifact_path: Path = Path("models/transformer/model")

    @property
    def label2id(self) -> dict[str, int]:
        return {"likely_fake": 0, "likely_real": 1}

    @property
    def id2label(self) -> dict[int, str]:
        return {value: key for key, value in self.label2id.items()}


__all__ = ["TransformerConfig"]
