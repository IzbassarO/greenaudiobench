"""Smoke-harness sampling must stay fold-aware (regression for the M3 smoke halt).

ESC-50 metadata is sorted by filename, whose prefix is the fold id, so a head
slice contains fold 1 only and the official-fold probe protocol cannot run.
The subset selector must cover every official fold instead — without touching
the production (non-smoke) selection or the probe protocol itself.
"""

import numpy as np
import pandas as pd
import pytest

from gab.datasets import (
    ESC50, SMOKE_CLIPS_PER_CLASS_PER_FOLD, SMOKE_MIN_CLASSES, smoke_subset,
)
from gab.probes import run_probe


def make_esc50_like(n_classes=6, per_class_per_fold=4):
    """Synthetic ESC-50-shaped metadata, sorted by filename like the real CSV."""
    rows = []
    for fold in ESC50.folds:
        for target in range(n_classes):
            for k in range(per_class_per_fold):
                src = 100000 + fold * 1000 + target * 10 + k
                rows.append({
                    "filename": f"{fold}-{src}-A-{target}.wav",
                    "fold": fold, "target": target, "category": f"class_{target}",
                    "esc10": False, "src_file": src, "take": "A",
                })
    return pd.DataFrame(rows).sort_values("filename").reset_index(drop=True)


def test_head_slice_would_cover_one_fold_only():
    """The bug being fixed: a prefix slice is single-fold."""
    meta = make_esc50_like()
    assert meta.iloc[:8]["fold"].nunique() == 1


def test_subset_covers_every_official_fold():
    sub = smoke_subset(make_esc50_like(), ESC50, 8)
    assert sorted(sub["fold"].unique()) == list(ESC50.folds)
    counts = sub["fold"].value_counts()
    assert counts.nunique() == 1  # fold-balanced


def test_every_fold_carries_at_least_two_classes():
    sub = smoke_subset(make_esc50_like(), ESC50, 8)
    for fold, group in sub.groupby("fold"):
        assert group["target"].nunique() >= SMOKE_MIN_CLASSES
        assert (group["target"].value_counts() == SMOKE_CLIPS_PER_CLASS_PER_FOLD).all()


def test_subset_is_deterministic():
    meta = make_esc50_like()
    a, b = smoke_subset(meta, ESC50, 8), smoke_subset(meta, ESC50, 8)
    assert a.equals(b)
    assert a["filename"].tolist() == b["filename"].tolist()


def test_subset_preserves_real_rows_and_metadata_order():
    meta = make_esc50_like()
    sub = smoke_subset(meta, ESC50, 8)
    assert list(sub.index) == sorted(sub.index)          # official metadata order
    merged = meta.set_index("filename").loc[sub["filename"]]
    np.testing.assert_array_equal(merged["fold"].to_numpy(), sub["fold"].to_numpy())
    np.testing.assert_array_equal(merged["target"].to_numpy(), sub["target"].to_numpy())


def test_target_n_is_a_minimum_that_expands():
    meta = make_esc50_like()
    small = smoke_subset(meta, ESC50, 8)
    bigger = smoke_subset(meta, ESC50, 40)
    assert len(small) >= 8 and len(bigger) >= 40
    assert bigger["target"].nunique() > small["target"].nunique()
    assert sorted(bigger["fold"].unique()) == list(ESC50.folds)


def test_subset_supports_the_real_probe_protocol():
    """The whole point: run_probe() must complete on a smoke subset, using the
    unmodified fold-aware inner CV."""
    sub = smoke_subset(make_esc50_like(), ESC50, 8)
    rng = np.random.default_rng(0)
    y = sub["target"].to_numpy()
    X = (rng.normal(size=(len(sub), 6)) + 5.0 * y[:, None]).astype(np.float32)
    res = run_probe(X, y, sub["fold"].to_numpy(), "esc50", "fake")
    assert list(res["outer_fold"]) == list(ESC50.folds)  # one row per official fold
    assert res["accuracy"].notna().all()


def test_refuses_when_no_class_spans_every_fold():
    meta = make_esc50_like(n_classes=2, per_class_per_fold=1)
    meta = meta[~((meta["fold"] == 3) & (meta["target"] == 1))]  # class 1 misses fold 3
    with pytest.raises(ValueError, match="fold-aware smoke subset"):
        smoke_subset(meta, ESC50, 8)
