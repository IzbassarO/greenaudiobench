"""M4 zero-shot scoring core: cosine invariance, prompts, per-fold rows."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_zeroshot as zs  # noqa: E402

from gab.datasets import ESC50  # noqa: E402


def test_cosine_scores_scale_invariant():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((6, 8)).astype(np.float32)
    T = rng.standard_normal((3, 8)).astype(np.float32)
    base = zs.cosine_scores(A, T).argmax(axis=1)
    scaled = A.copy()
    scaled[0] *= 100.0  # if L2 normalization were missing, row 0 would dominate
    np.testing.assert_array_equal(zs.cosine_scores(scaled, T).argmax(axis=1), base)
    np.testing.assert_array_equal(zs.cosine_scores(A, T * 42.0).argmax(axis=1), base)


def test_l2_normalize_rejects_zero_rows():
    with pytest.raises(ValueError, match="zero-norm"):
        zs.l2_normalize(np.zeros((2, 4), dtype=np.float32))


def test_class_names_underscores_and_order():
    meta = pd.DataFrame({
        "target": [1, 0, 2, 1],
        "category": ["chirping_birds", "dog", "sea_waves", "chirping_birds"],
    })
    names = zs.class_names_by_id(meta, ESC50)
    assert names == ["dog", "chirping birds", "sea waves"]
    prompts = zs.build_prompts(zs.TEMPLATES["primary"], names)
    assert prompts[1] == "a sound of chirping birds"


def test_class_names_reject_non_contiguous_ids():
    meta = pd.DataFrame({"target": [0, 5], "category": ["dog", "rain"]})
    with pytest.raises(ValueError, match="contiguous"):
        zs.class_names_by_id(meta, ESC50)


def test_per_fold_metrics_one_row_per_official_fold():
    preds = np.array([0, 1, 0, 1, 1, 0])
    labels = np.array([0, 1, 1, 1, 0, 0])
    folds = np.array([1, 1, 2, 2, 3, 3])
    df = zs.per_fold_metrics(preds, labels, folds, n_classes=2)
    assert list(df["fold"]) == [1, 2, 3]
    assert list(df["n_clips"]) == [2, 2, 2]
    assert df.loc[0, "accuracy"] == 1.0
    assert df.loc[1, "accuracy"] == 0.5
    assert df.loc[2, "accuracy"] == 0.5


def test_end_to_end_scoring_with_fake_embeddings():
    # class-k clips point along axis k -> argmax must recover the label
    n_classes, clips_per_class = 4, 3
    A = np.repeat(np.eye(n_classes, 8, dtype=np.float32), clips_per_class, axis=0)
    A += 0.01  # small offset, still nearest to own prompt
    T = np.eye(n_classes, 8, dtype=np.float32)
    preds = zs.cosine_scores(A, T).argmax(axis=1)
    labels = np.repeat(np.arange(n_classes), clips_per_class)
    np.testing.assert_array_equal(preds, labels)
