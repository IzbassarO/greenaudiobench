"""Official cross-validation fold handling for ESC-50 and UrbanSound8K.

Hard rule: official folds only — never re-split, never merge. The loaders
here validate the official metadata files strictly and refuse anything that
deviates from the published fold structure.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd

ESC50_FOLDS: tuple[int, ...] = (1, 2, 3, 4, 5)
US8K_FOLDS: tuple[int, ...] = tuple(range(1, 11))

ESC50_REQUIRED_COLUMNS = {"filename", "fold", "target", "category"}
US8K_REQUIRED_COLUMNS = {"slice_file_name", "fold", "classID", "class"}


class MetadataError(ValueError):
    """Raised when a dataset metadata file violates the official structure."""


def _check_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise MetadataError(f"{name}: missing required columns {sorted(missing)}")


def _check_folds(df: pd.DataFrame, expected: tuple[int, ...], name: str) -> None:
    found = set(df["fold"].unique())
    if found != set(expected):
        raise MetadataError(
            f"{name}: fold set {sorted(found)} != official folds {list(expected)}"
        )


def _check_unique(df: pd.DataFrame, column: str, name: str) -> None:
    dupes = df[column][df[column].duplicated()].unique()
    if len(dupes) > 0:
        raise MetadataError(f"{name}: duplicate {column} entries, e.g. {dupes[:3]}")


def _check_bijection(df: pd.DataFrame, id_col: str, label_col: str, name: str) -> None:
    """Every numeric label id must map to exactly one class name and vice versa."""
    if df.groupby(id_col)[label_col].nunique().max() > 1:
        raise MetadataError(f"{name}: some {id_col} maps to multiple {label_col} values")
    if df.groupby(label_col)[id_col].nunique().max() > 1:
        raise MetadataError(f"{name}: some {label_col} maps to multiple {id_col} values")


def load_esc50_meta(csv_path: str | Path) -> pd.DataFrame:
    """Load and strictly validate the official ESC-50 metadata (meta/esc50.csv)."""
    df = pd.read_csv(csv_path)
    _check_columns(df, ESC50_REQUIRED_COLUMNS, "esc50")
    _check_folds(df, ESC50_FOLDS, "esc50")
    _check_unique(df, "filename", "esc50")
    _check_bijection(df, "target", "category", "esc50")
    # ESC-50 filenames are "{fold}-{src_file}-{take}-{target}.wav"; the prefix
    # must agree with the fold column — a cheap guard against corrupted metadata.
    prefix_fold = df["filename"].str.split("-").str[0].astype(int)
    if not (prefix_fold == df["fold"]).all():
        bad = df.loc[prefix_fold != df["fold"], "filename"].head(3).tolist()
        raise MetadataError(f"esc50: filename fold prefix disagrees with fold column, e.g. {bad}")
    return df


def load_us8k_meta(csv_path: str | Path) -> pd.DataFrame:
    """Load and strictly validate the official UrbanSound8K metadata."""
    df = pd.read_csv(csv_path)
    _check_columns(df, US8K_REQUIRED_COLUMNS, "us8k")
    _check_folds(df, US8K_FOLDS, "us8k")
    _check_unique(df, "slice_file_name", "us8k")
    _check_bijection(df, "classID", "class", "us8k")
    return df


def iter_official_folds(
    meta: pd.DataFrame, fold_column: str = "fold"
) -> Iterator[tuple[int, pd.DataFrame, pd.DataFrame]]:
    """Yield (test_fold, train_df, test_df) for each official fold, in order.

    The test set of fold k is exactly the rows with fold == k; the train set is
    everything else. No re-splitting, no merging, no shuffling.
    """
    folds = sorted(meta[fold_column].unique())
    for fold in folds:
        test_mask = meta[fold_column] == fold
        test_df = meta[test_mask]
        train_df = meta[~test_mask]
        assert len(train_df) + len(test_df) == len(meta)
        assert not set(train_df.index) & set(test_df.index)
        yield int(fold), train_df, test_df
