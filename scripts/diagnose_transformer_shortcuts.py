"""Measure source-style markers and paired cue sensitivity of DistilBERT."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MARKERS = {
    "reuters": re.compile(r"\breuters\b", re.I),
    "associated_press": re.compile(r"\bassociated\s+press\b|\bthe\s+ap\b", re.I),
    "afp": re.compile(r"\bafp\b|\bagence\s+france-presse\b", re.I),
    "wire_dateline_prefix": re.compile(
        r"^\s*(?:\([A-Z][A-Z .-]{1,30}\)\s*[-:]|[A-Z][A-Z .-]{2,30}\s[-:])"
    ),
}

NASA_TEXT = (
    "NASA's Perseverance rover landed on Mars in February 2021. The rover is "
    "exploring Jezero Crater, a site scientists selected because it may preserve "
    "evidence of ancient microbial life. NASA says the mission is collecting "
    "rock and soil samples for possible return to Earth in a future campaign."
)
PAIRED_VARIANTS = (
    ("plain", NASA_TEXT),
    ("reuters_dateline", f"WASHINGTON (Reuters) - {NASA_TEXT}"),
    ("reuters_parenthetical", f"(Reuters) {NASA_TEXT}"),
    ("ap_dateline", f"WASHINGTON (AP) - {NASA_TEXT}"),
    ("nasa_agency_prefix", f"NASA reports: {NASA_TEXT}"),
    ("reuters_signature", f"{NASA_TEXT} (Reuters)"),
)


def inspect_isot(raw_dir: Path) -> dict[str, object]:
    results: dict[str, object] = {}
    for label, filename in (("likely_fake", "Fake.csv"), ("likely_real", "True.csv")):
        frame = pd.read_csv(raw_dir / filename, encoding="utf-8")
        texts = frame["text"].fillna("").astype(str)
        results[label] = {
            "rows": len(frame),
            "marker_rates": {
                name: float(
                    texts.map(lambda value, marker=pattern: bool(marker.search(value))).mean()
                )
                for name, pattern in MARKERS.items()
            },
            "mean_body_characters": float(texts.str.len().mean()),
        }
    return results


def paired_predictions(artifact: Path, device: str | None) -> list[dict[str, object]]:
    tokenizer = AutoTokenizer.from_pretrained(artifact)
    model = AutoModelForSequenceClassification.from_pretrained(artifact)
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(resolved_device).eval()
    labels = {int(index): label.lower() for index, label in model.config.id2label.items()}
    encoded = tokenizer(
        [text for _, text in PAIRED_VARIANTS],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    )
    encoded = {key: value.to(resolved_device) for key, value in encoded.items()}
    with torch.inference_mode():
        probabilities = torch.softmax(model(**encoded).logits, dim=-1).cpu().tolist()
    fake_index = next(index for index, label in labels.items() if label == "likely_fake")
    reports = []
    for (variant, _), row in zip(PAIRED_VARIANTS, probabilities, strict=True):
        predicted_index = max(range(len(row)), key=row.__getitem__)
        reports.append(
            {
                "variant": variant,
                "predicted_label": labels[predicted_index],
                "max_probability": float(row[predicted_index]),
                "likely_fake_probability": float(row[fake_index]),
            }
        )
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("datasets/raw/isot"))
    parser.add_argument("--artifact", type=Path, default=Path("models/transformer/distilbert"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/experiments/transformer-shortcut-diagnostic.json"),
    )
    parser.add_argument("--device")
    args = parser.parse_args()
    report = {
        "diagnostic": "source_style_shortcut_probe",
        "warning": (
            "Marker correlation and a paired probe indicate sensitivity, not causal "
            "proof or a generalization benchmark."
        ),
        "dataset": "ISOT",
        "source_marker_rates_by_label": inspect_isot(args.raw_dir),
        "paired_nasa_variants": paired_predictions(args.artifact, args.device),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()
