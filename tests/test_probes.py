"""C1/C2: deterministic probe, one row per official outer fold, fold-aware inner CV."""

import numpy as np
import pytest

from gab.probes import C_GRID, run_probe, select_C_inner_cv


def make_blobs(n_folds=5, per_fold=20, dim=8, n_classes=2, seed=0):
    rng = np.random.default_rng(seed)
    X, y, folds = [], [], []
    for f in range(1, n_folds + 1):
        for c in range(n_classes):
            pts = rng.normal(loc=3.0 * c, scale=1.0, size=(per_fold // n_classes, dim))
            X.append(pts)
            y += [c] * (per_fold // n_classes)
            folds += [f] * (per_fold // n_classes)
    return np.vstack(X).astype(np.float32), np.array(y), np.array(folds)


def test_one_row_per_official_outer_fold():
    X, y, folds = make_blobs()
    res = run_probe(X, y, folds, "synthetic", "fake")
    assert list(res["outer_fold"]) == [1, 2, 3, 4, 5]
    assert (res["n_test"] == 20).all() and (res["n_train"] == 80).all()
    assert (res["accuracy"] > 0.9).all()  # trivially separable blobs


def test_probe_is_deterministic():
    X, y, folds = make_blobs()
    a = run_probe(X, y, folds, "synthetic", "fake")
    b = run_probe(X, y, folds, "synthetic", "fake")
    assert a.equals(b)


def test_inner_cv_rotates_only_official_training_folds():
    X, y, folds = make_blobs()
    train_mask = folds != 5
    best_C, scores = select_C_inner_cv(X[train_mask], y[train_mask], folds[train_mask])
    assert best_C in C_GRID and set(scores) == set(C_GRID)
    # PredefinedSplit must produce exactly one inner split per official train fold
    from sklearn.model_selection import PredefinedSplit

    splits = list(PredefinedSplit(test_fold=folds[train_mask]).split())
    assert len(splits) == 4  # folds 1-4 each serve as inner validation once
    for inner_train, inner_val in splits:
        val_folds = set(folds[train_mask][inner_val])
        assert len(val_folds) == 1  # inner val = exactly one whole official fold
        assert val_folds.isdisjoint(set(folds[train_mask][inner_train]))


def test_inner_cv_requires_multiple_folds():
    X, y, folds = make_blobs(n_folds=1)
    with pytest.raises(ValueError, match="at least 2"):
        select_C_inner_cv(X, y, folds)


def test_probe_rejects_nan_embeddings():
    X, y, folds = make_blobs()
    X[3, 2] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        run_probe(X, y, folds, "synthetic", "fake")


def test_probe_rejects_misaligned_inputs():
    X, y, folds = make_blobs()
    with pytest.raises(ValueError, match="aligned"):
        run_probe(X, y[:-1], folds, "synthetic", "fake")
