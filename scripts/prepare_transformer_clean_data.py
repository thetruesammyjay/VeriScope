"""Create marker-cleaned Transformer-only copies of existing ISOT splits."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from ml.transformer.cleaning import clean_article_text
from sklearn.model_selection import train_test_split


def prepare(input_dir: Path, output_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        frame = pd.read_csv(input_dir / f"{split}.csv")
        if "text" not in frame or "label" not in frame:
            raise ValueError(f"{split}.csv must contain text and label columns")
        frame["text"] = frame["text"].astype(str).map(clean_article_text)
        counts[split] = len(frame)
        if split == "validation":
            selection, calibration = train_test_split(
                frame,
                test_size=0.5,
                random_state=42,
                stratify=frame["label"],
            )
            splits = {
                "model_selection": selection,
                "calibration": calibration,
            }
        else:
            splits = {split: frame}
        output_dir.mkdir(parents=True, exist_ok=True)
        for split_name, split_frame in splits.items():
            split_frame.to_csv(output_dir / f"{split_name}.csv", index=False)
            counts[split_name] = len(split_frame)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("datasets/processed"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/processed/transformer_source_clean"),
    )
    args = parser.parse_args()
    for split, count in prepare(args.input_dir, args.output_dir).items():
        print(f"Prepared {count} {split} rows in {args.output_dir}")


if __name__ == "__main__":
    main()
