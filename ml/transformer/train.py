"""Fine-tune and persist a Hugging Face sequence classifier."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .config import TransformerConfig
from .dataset import load_frame, tokenize_frame


def compute_metrics(eval_prediction: Any) -> dict[str, float]:
    """Return the metrics used for transformer model selection."""

    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_macro": float(
            precision_score(labels, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(labels, predictions, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
    }


def _training_arguments(
    config: TransformerConfig, *, has_eval: bool
) -> TrainingArguments:
    """Build version-compatible TrainingArguments."""

    values: dict[str, Any] = {
        "output_dir": str(config.artifact_path),
        "per_device_train_batch_size": config.train_batch_size,
        "per_device_eval_batch_size": config.eval_batch_size,
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.num_train_epochs,
        "weight_decay": config.weight_decay,
        "logging_steps": config.logging_steps,
        "seed": config.random_state,
        "report_to": [],
    }
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    if "warmup_ratio" in parameters:
        values["warmup_ratio"] = config.warmup_ratio
    elif "warmup_steps" in parameters:
        # Transformers 5 removed warmup_ratio; callers can still override this
        # with a future training-argument adapter without changing the config.
        values["warmup_steps"] = 0
    if has_eval and config.save_checkpoints:
        values["save_strategy"] = "epoch"
        evaluation_key = (
            "eval_strategy" if "eval_strategy" in parameters else "evaluation_strategy"
        )
        values[evaluation_key] = "epoch"
        values["load_best_model_at_end"] = True
        values["metric_for_best_model"] = "f1_macro"
        values["greater_is_better"] = True
    else:
        values["save_strategy"] = "no"
    return TrainingArguments(**values)


def train_csv(
    train_csv: Path,
    *,
    validation_csv: Path | None = None,
    config: TransformerConfig | None = None,
    text_column: str = "text",
    label_column: str = "label",
) -> Path:
    """Fine-tune a model from CSV files and save it with its tokenizer."""

    resolved = config or TransformerConfig()
    set_seed(resolved.random_state)
    train_frame = load_frame(
        train_csv, text_column=text_column, label_column=label_column
    )
    eval_frame = (
        load_frame(validation_csv, text_column=text_column, label_column=label_column)
        if validation_csv
        else None
    )
    tokenizer = AutoTokenizer.from_pretrained(resolved.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        resolved.model_name,
        num_labels=2,
        id2label=resolved.id2label,
        label2id=resolved.label2id,
    )
    train_dataset = tokenize_frame(train_frame, tokenizer, config=resolved)
    eval_dataset = (
        tokenize_frame(eval_frame, tokenizer, config=resolved)
        if eval_frame is not None
        else None
    )
    trainer_values: dict[str, Any] = {
        "model": model,
        "args": _training_arguments(resolved, has_eval=eval_dataset is not None),
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics if eval_dataset is not None else None,
    }
    trainer_parameters = inspect.signature(Trainer.__init__).parameters
    tokenizer_key = "processing_class" if "processing_class" in trainer_parameters else "tokenizer"
    trainer_values[tokenizer_key] = tokenizer
    trainer = Trainer(**trainer_values)
    trainer.train()
    resolved.artifact_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(resolved.artifact_path)
    tokenizer.save_pretrained(resolved.artifact_path)
    metadata = {
        "model": "transformer_sequence_classifier",
        "model_version": resolved.model_version,
        "base_model": resolved.model_name,
        "label_mapping": resolved.label2id,
        "max_length": resolved.max_length,
    }
    (resolved.artifact_path / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return resolved.artifact_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_csv", type=Path)
    parser.add_argument("--validation-csv", type=Path)
    parser.add_argument("--model-name", default=TransformerConfig().model_name)
    parser.add_argument("--epochs", type=float, default=TransformerConfig().num_train_epochs)
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument(
        "--artifact-path", type=Path, default=TransformerConfig().artifact_path
    )
    args = parser.parse_args()
    path = train_csv(
        args.train_csv,
        validation_csv=args.validation_csv,
        config=TransformerConfig(
            model_name=args.model_name,
            artifact_path=args.artifact_path,
            num_train_epochs=args.epochs,
            save_checkpoints=args.save_checkpoints,
        ),
    )
    print(f"Saved transformer model artifact to {path}")


if __name__ == "__main__":
    main()


__all__ = ["compute_metrics", "main", "train_csv"]
