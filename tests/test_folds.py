"""Unit tests for official fold parsing — tiny synthetic fixtures, no audio."""

import pandas as pd
import pytest

from gab.folds import (
    ESC50_FOLDS,
    US8K_FOLDS,
    MetadataError,
    iter_official_folds,
    load_esc50_meta,
    load_us8k_meta,
)

CATEGORIES = {0: "dog", 1: "rain"}
US8K_CLASSES = {0: "air_conditioner", 1: "car_horn"}


def make_esc50_df(n_per_fold=2):
    rows = []
    for fold in ESC50_FOLDS:
        for i in range(n_per_fold):
            target = i % 2
            rows.append(
                {
                    "filename": f"{fold}-{100000 + fold * 10 + i}-A-{target}.wav",
                    "fold": fold,
                    "target": target,
                    "category": CATEGORIES[target],
                    "esc10": True,
                    "src_file": 100000 + fold * 10 + i,
                    "take": "A",
                }
            )
    return pd.DataFrame(rows)


def make_us8k_df(n_per_fold=1):
    rows = []
    for fold in US8K_FOLDS:
        for i in range(n_per_fold):
            class_id = (fold + i) % 2
            rows.append(
                {
                    "slice_file_name": f"{1000 + fold * 10 + i}-{class_id}-0-0.wav",
                    "fsID": 1000 + fold * 10 + i,
                    "start": 0.0,
                    "end": 4.0,
                    "salience": 1,
                    "fold": fold,
                    "classID": class_id,
                    "class": US8K_CLASSES[class_id],
                }
            )
    return pd.DataFrame(rows)


def write_csv(df, tmp_path, name):
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------- ESC-50


def test_esc50_loads_valid_meta(tmp_path):
    df = load_esc50_meta(write_csv(make_esc50_df(), tmp_path, "esc50.csv"))
    assert len(df) == 10
    assert set(df["fold"]) == set(ESC50_FOLDS)


def test_esc50_official_fold_iteration(tmp_path):
    df = load_esc50_meta(write_csv(make_esc50_df(), tmp_path, "esc50.csv"))
    seen_folds = []
    for fold, train, test in iter_official_folds(df):
        seen_folds.append(fold)
        # test set is EXACTLY the rows of this fold — no re-splitting
        assert (test["fold"] == fold).all()
        assert (train["fold"] != fold).all()
        assert len(train) + len(test) == len(df)
        assert not set(train["filename"]) & set(test["filename"])
    assert seen_folds == list(ESC50_FOLDS)


def test_esc50_rejects_missing_column(tmp_path):
    bad = make_esc50_df().drop(columns=["fold"])
    with pytest.raises(MetadataError, match="missing required columns"):
        load_esc50_meta(write_csv(bad, tmp_path, "esc50.csv"))


def test_esc50_rejects_unexpected_fold(tmp_path):
    bad = make_esc50_df()
    bad.loc[0, "fold"] = 6
    bad.loc[0, "filename"] = "6-100010-A-0.wav"
    with pytest.raises(MetadataError, match="official folds"):
        load_esc50_meta(write_csv(bad, tmp_path, "esc50.csv"))


def test_esc50_rejects_incomplete_folds(tmp_path):
    bad = make_esc50_df()
    bad = bad[bad["fold"] != 5]
    with pytest.raises(MetadataError, match="official folds"):
        load_esc50_meta(write_csv(bad, tmp_path, "esc50.csv"))


def test_esc50_rejects_duplicate_filenames(tmp_path):
    bad = make_esc50_df()
    bad.loc[1, "filename"] = bad.loc[0, "filename"]
    with pytest.raises(MetadataError, match="duplicate"):
        load_esc50_meta(write_csv(bad, tmp_path, "esc50.csv"))


def test_esc50_rejects_label_mapping_conflict(tmp_path):
    bad = make_esc50_df()
    bad.loc[0, "category"] = "cat"  # target 0 now maps to both "dog" and "cat"
    with pytest.raises(MetadataError, match="maps to multiple"):
        load_esc50_meta(write_csv(bad, tmp_path, "esc50.csv"))


def test_esc50_rejects_fold_prefix_mismatch(tmp_path):
    bad = make_esc50_df()
    bad.loc[0, "filename"] = "2-999999-A-0.wav"  # row says fold 1, name says fold 2
    with pytest.raises(MetadataError, match="fold prefix"):
        load_esc50_meta(write_csv(bad, tmp_path, "esc50.csv"))


# ------------------------------------------------------------ UrbanSound8K


def test_us8k_loads_valid_meta(tmp_path):
    df = load_us8k_meta(write_csv(make_us8k_df(), tmp_path, "us8k.csv"))
    assert len(df) == 10
    assert set(df["fold"]) == set(US8K_FOLDS)


def test_us8k_official_fold_iteration(tmp_path):
    df = load_us8k_meta(write_csv(make_us8k_df(n_per_fold=2), tmp_path, "us8k.csv"))
    seen_folds = []
    for fold, train, test in iter_official_folds(df):
        seen_folds.append(fold)
        assert (test["fold"] == fold).all()
        assert (train["fold"] != fold).all()
        assert len(train) + len(test) == len(df)
    assert seen_folds == list(US8K_FOLDS)


def test_us8k_rejects_missing_column(tmp_path):
    bad = make_us8k_df().drop(columns=["classID"])
    with pytest.raises(MetadataError, match="missing required columns"):
        load_us8k_meta(write_csv(bad, tmp_path, "us8k.csv"))


def test_us8k_rejects_unexpected_fold(tmp_path):
    bad = make_us8k_df()
    bad.loc[0, "fold"] = 11
    with pytest.raises(MetadataError, match="official folds"):
        load_us8k_meta(write_csv(bad, tmp_path, "us8k.csv"))


def test_us8k_rejects_incomplete_folds(tmp_path):
    bad = make_us8k_df()
    bad = bad[bad["fold"] != 10]
    with pytest.raises(MetadataError, match="official folds"):
        load_us8k_meta(write_csv(bad, tmp_path, "us8k.csv"))


def test_us8k_rejects_duplicate_filenames(tmp_path):
    bad = make_us8k_df(n_per_fold=2)
    bad.loc[1, "slice_file_name"] = bad.loc[0, "slice_file_name"]
    with pytest.raises(MetadataError, match="duplicate"):
        load_us8k_meta(write_csv(bad, tmp_path, "us8k.csv"))


def test_us8k_rejects_label_mapping_conflict(tmp_path):
    bad = make_us8k_df()
    bad.loc[0, "class"] = "engine_idling"
    with pytest.raises(MetadataError, match="maps to multiple"):
        load_us8k_meta(write_csv(bad, tmp_path, "us8k.csv"))
