"""Create deterministic, stratified train/validation/test splits."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .validate import canonical_label, validate_dataset

SPLIT_NAMES = ("train", "validation", "test")


def _validate_fractions(
    train_size: float, validation_size: float, test_size: float
) -> None:
    values = (train_size, validation_size, test_size)
    if any(value <= 0 or value >= 1 for value in values):
        raise ValueError("split fractions must be greater than 0 and less than 1")
    if not np.isclose(sum(values), 1.0):
        raise ValueError("train, validation, and test fractions must sum to 1")


def _allocate_counts(size: int, fractions: tuple[float, float, float]) -> list[int]:
    """Allocate a class's rows using largest remainders."""

    if size < len(fractions):
        raise ValueError(
            "each label needs at least one row in train, validation, and test"
        )
    exact = np.asarray(fractions) * size
    counts = np.floor(exact).astype(int)
    for index in np.argsort(-(exact - counts))[: size - int(counts.sum())]:
        counts[index] += 1
    for index in range(len(counts)):
        if counts[index] == 0:
            donor = int(np.argmax(counts))
            counts[donor] -= 1
            counts[index] += 1
    return counts.tolist()


def split_dataframe(
    frame: pd.DataFrame,
    *,
    train_size: float = 0.8,
    validation_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    """Return reproducible, label-stratified dataframes with no overlap."""

    _validate_fractions(train_size, validation_size, test_size)
    working = frame.reset_index(drop=True).copy()
    working["label"] = working["label"].map(canonical_label)
    validate_dataset(working)
    fractions = (train_size, validation_size, test_size)
    rng = np.random.default_rng(random_state)
    buckets: dict[str, list[int]] = {name: [] for name in SPLIT_NAMES}

    for label in sorted(working["label"].unique()):
        indices = working.index[working["label"] == label].to_numpy(copy=True)
        rng.shuffle(indices)
        counts = _allocate_counts(len(indices), fractions)
        start = 0
        for name, count in zip(SPLIT_NAMES, counts):
            buckets[name].extend(indices[start : start + count].tolist())
            start += count

    result: dict[str, pd.DataFrame] = {}
    for name in SPLIT_NAMES:
        indices = buckets[name]
        rng.shuffle(indices)
        result[name] = working.loc[indices].reset_index(drop=True)
    return result


def temporal_split_dataframe(
    frame: pd.DataFrame,
    *,
    date_column: str = "date",
    train_size: float = 0.8,
    validation_size: float = 0.1,
    test_size: float = 0.1,
) -> dict[str, pd.DataFrame]:
    """Split chronologically so future articles never enter training."""

    _validate_fractions(train_size, validation_size, test_size)
    working = frame.reset_index(drop=True).copy()
    if date_column not in working:
        raise ValueError(f"Temporal splitting requires a {date_column!r} column")
    working["label"] = working["label"].map(canonical_label)
    validate_dataset(working)
    dates = pd.to_datetime(working[date_column], errors="coerce", format="mixed")
    invalid_dates = dates.isna()
    if invalid_dates.any():
        warnings.warn(
            f"Excluding {int(invalid_dates.sum())} rows with unparseable "
            f"{date_column!r} values from temporal splits.",
            UserWarning,
            stacklevel=2,
        )
        working = working.loc[~invalid_dates].reset_index(drop=True)
        dates = dates.loc[~invalid_dates].reset_index(drop=True)
    ordered = working.assign(_parsed_date=dates).sort_values(
        ["_parsed_date"], kind="mergesort"
    )
    counts = _allocate_counts(len(ordered), (train_size, validation_size, test_size))
    train_end = counts[0]
    validation_end = train_end + counts[1]
    return {
        "train": ordered.iloc[:train_end].drop(columns=["_parsed_date"]).reset_index(
            drop=True
        ),
        "validation": ordered.iloc[train_end:validation_end]
        .drop(columns=["_parsed_date"])
        .reset_index(drop=True),
        "test": ordered.iloc[validation_end:]
        .drop(columns=["_parsed_date"])
        .reset_index(drop=True),
    }


def save_splits(
    splits: dict[str, pd.DataFrame], output_dir: Path
) -> dict[str, Path]:
    """Write split CSV files and return their paths."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name in SPLIT_NAMES:
        if name not in splits:
            raise ValueError(f"Missing required split: {name}")
        path = output_dir / f"{name}.csv"
        splits[name].to_csv(path, index=False)
        paths[name] = path
    return paths


def split_dataset(
    input_path: Path = Path("datasets/processed/dataset.csv"),
    output_dir: Path = Path("datasets/processed"),
    *,
    train_size: float = 0.8,
    validation_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
    strategy: str = "random",
    date_column: str = "date",
) -> dict[str, Path]:
    """Read a processed CSV, split it, and save train/validation/test files."""

    frame = pd.read_csv(input_path)
    if strategy == "random":
        splits = split_dataframe(
            frame,
            train_size=train_size,
            validation_size=validation_size,
            test_size=test_size,
            random_state=random_state,
        )
    elif strategy == "temporal":
        splits = temporal_split_dataframe(
            frame,
            date_column=date_column,
            train_size=train_size,
            validation_size=validation_size,
            test_size=test_size,
        )
    else:
        raise ValueError("strategy must be either 'random' or 'temporal'")
    return save_splits(splits, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("datasets/processed/dataset.csv")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("datasets/processed")
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--strategy", choices=("random", "temporal"), default="random"
    )
    parser.add_argument("--date-column", default="date")
    args = parser.parse_args()
    paths = split_dataset(
        args.input,
        args.output_dir,
        random_state=args.random_state,
        strategy=args.strategy,
        date_column=args.date_column,
    )
    for name, path in paths.items():
        print(f"Saved {name} split to {path}")


if __name__ == "__main__":
    main()


__all__ = [
    "SPLIT_NAMES",
    "main",
    "save_splits",
    "split_dataframe",
    "split_dataset",
    "temporal_split_dataframe",
]
