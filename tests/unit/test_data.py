from pathlib import Path

import pandas as pd
import pytest

from ml.data.build_dataset import prepare_dataset
from ml.data.load_isot import load_isot_dataset
from ml.data.split import split_dataframe, temporal_split_dataframe
from ml.data.validate import DatasetValidationError, validate_dataset


def _write_isot_files(directory: Path) -> None:
    directory.mkdir(parents=True)
    pd.DataFrame(
        {
            "title": ["Fake headline"],
            "text": ["A fabricated report."],
            "subject": ["news"],
            "date": ["2020-01-01"],
        }
    ).to_csv(directory / "Fake.csv", index=False)
    pd.DataFrame(
        {
            "title": ["Real headline"],
            "text": ["An official report."],
            "subject": ["politics"],
            "date": ["2020-01-02"],
        }
    ).to_csv(directory / "True.csv", index=False)


def test_load_isot_adds_canonical_labels(tmp_path: Path):
    raw_dir = tmp_path / "isot"
    _write_isot_files(raw_dir)

    frame = load_isot_dataset(raw_dir)

    assert len(frame) == 2
    assert set(frame["label"]) == {"likely_fake", "likely_real"}
    assert set(frame["source_file"]) == {"Fake.csv", "True.csv"}


def test_prepare_dataset_combines_title_and_body_and_deduplicates():
    frame = pd.DataFrame(
        {
            "title": ["Same title", " Same   title ", "Same title"],
            "text": ["Same body", "same body", "same body"],
            "label": ["real", "real", "fake"],
        }
    )

    prepared = prepare_dataset(frame)

    # The repeated normalized text is removed, including its conflicting copy.
    assert len(prepared) == 0

    same_label = prepare_dataset(frame.iloc[:2])
    assert len(same_label) == 1


def test_validate_dataset_rejects_duplicate_text():
    frame = pd.DataFrame(
        {
            "text": ["A report was published.", " a report was published. "],
            "label": ["likely_real", "likely_real"],
        }
    )

    with pytest.raises(DatasetValidationError, match="duplicate"):
        validate_dataset(frame)


def test_split_dataframe_is_stratified_reproducible_and_disjoint():
    frame = pd.DataFrame(
        {
            "text": [f"Article {index}" for index in range(20)],
            "label": ["likely_fake"] * 10 + ["likely_real"] * 10,
        }
    )

    first = split_dataframe(frame, random_state=7)
    second = split_dataframe(frame, random_state=7)

    assert {name: len(value) for name, value in first.items()} == {
        "train": 16,
        "validation": 2,
        "test": 2,
    }
    assert all(first[name].equals(second[name]) for name in first)
    assert set(first["train"]["text"]).isdisjoint(first["test"]["text"])
    assert set(first["train"]["text"]).isdisjoint(first["validation"]["text"])
    assert set(first["validation"]["text"]).isdisjoint(first["test"]["text"])
    assert all(first[name]["label"].nunique() == 2 for name in first)


def test_temporal_split_keeps_future_dates_out_of_training():
    frame = pd.DataFrame(
        {
            "text": [f"Article {index}" for index in range(12)],
            "label": ["likely_fake", "likely_real"] * 6,
            "date": pd.date_range("2020-01-01", periods=12, freq="D").astype(str),
        }
    )

    splits = temporal_split_dataframe(frame)

    assert max(splits["train"]["date"]) < min(splits["validation"]["date"])
    assert max(splits["validation"]["date"]) < min(splits["test"]["date"])


def test_temporal_split_excludes_unparseable_dates_with_warning():
    frame = pd.DataFrame(
        {
            "text": [f"Article {index}" for index in range(12)],
            "label": ["likely_fake", "likely_real"] * 6,
            "date": ["not-a-date"] + list(
                pd.date_range("2020-01-01", periods=11, freq="D").astype(str)
            ),
        }
    )

    with pytest.warns(UserWarning, match="unparseable"):
        splits = temporal_split_dataframe(frame)

    assert sum(len(split) for split in splits.values()) == 11
