"""Embedding cache: round-trip, alignment guarantee, metadata-keyed cache hits."""

import numpy as np
import pandas as pd
import pytest

from gab import cache as cache_mod
from gab.cache import is_cached, load_embeddings, save_embeddings, verify_alignment
from gab.datasets import ESC50


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_mod, "REPO_ROOT", tmp_path)
    return tmp_path


def _save(fake_repo, filenames, meta_extra=None):
    X = np.arange(len(filenames) * 4, dtype=np.float32).reshape(len(filenames), 4)
    save_embeddings(
        cache_mod.cache_dir("fake", "esc50"),
        X, filenames,
        fold_ids=np.ones(len(filenames), dtype=int),
        label_ids=np.zeros(len(filenames), dtype=int),
        metadata={"model": "fake", "revision": "r1", **(meta_extra or {})},
    )
    return X


def test_round_trip_and_alignment(fake_repo):
    names = ["1-a-A-0.wav", "1-b-A-0.wav", "1-c-A-0.wav"]
    X = _save(fake_repo, names)
    X2, filenames, fold_ids, label_ids, meta = load_embeddings("fake", "esc50")
    np.testing.assert_array_equal(X, X2)
    assert filenames == names
    assert meta["n_clips"] == 3 and meta["embed_dim"] == 4
    assert "run" in meta and "git_commit" in meta["run"]

    meta_df = pd.DataFrame({ESC50.filename_column: names})
    verify_alignment(filenames, meta_df, ESC50)  # must not raise
    shuffled = pd.DataFrame({ESC50.filename_column: names[::-1]})
    with pytest.raises(ValueError, match="NOT aligned"):
        verify_alignment(filenames, shuffled, ESC50)


def test_cache_hit_is_metadata_keyed(fake_repo):
    _save(fake_repo, ["1-a-A-0.wav"])
    assert is_cached("fake", "esc50", {"model": "fake", "revision": "r1"})
    # different checkpoint revision -> NOT a cache hit, must re-extract
    assert not is_cached("fake", "esc50", {"model": "fake", "revision": "r2"})
    assert not is_cached("other", "esc50", {"model": "other"})


def test_save_rejects_misaligned_arrays(fake_repo):
    with pytest.raises(ValueError, match="aligned"):
        save_embeddings(
            cache_mod.cache_dir("fake", "esc50"),
            np.zeros((3, 4), dtype=np.float32), ["a.wav", "b.wav"],
            np.ones(3, dtype=int), np.zeros(3, dtype=int), {},
        )
