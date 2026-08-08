"""M2 extraction pipeline on a tiny synthetic dataset with a FakeAdapter.

Protects: metadata-row alignment, official-fold/label propagation, NaN and
shape assertions, metadata-keyed idempotency — without any model download.
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from gab import cache as cache_mod
from gab.cache import load_embeddings, verify_alignment
from gab.datasets import ESC50
from gab.models.base import AdapterInfo, EmbeddingAdapter

extract = importlib.import_module("extract_embeddings") if "extract_embeddings" in sys.modules else None
if extract is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import extract_embeddings as extract  # noqa: E402


class FakeAdapter(EmbeddingAdapter):
    """Deterministic embedding = [mean, std, len, const] of the waveform."""

    def __init__(self):
        self.info = AdapterInfo(
            name="fake", checkpoint="none", source_url="none", revision="r-test",
            sample_rate=8000, duration_policy="crop_or_pad:5.0s",
            preprocess_id="fake-v1", embed_dim=4, embedding_layer="stats",
        )

    def load(self, device="cpu", fp16=False):
        pass

    def embed_batch(self, wavs):
        out = np.stack([
            np.array([w.mean(), w.std(), len(w), 1.0], dtype=np.float32) for w in wavs
        ])
        return out


@pytest.fixture
def tiny_esc50(tmp_path, monkeypatch):
    """5 tiny wavs (one per official fold) + strictly valid esc50 meta."""
    root = tmp_path / "data" / "raw" / "esc50" / "ESC-50-master"
    (root / "audio").mkdir(parents=True)
    (root / "meta").mkdir()
    rows = []
    rng = np.random.default_rng(0)
    for fold in range(1, 6):
        target = fold % 2
        name = f"{fold}-{100000 + fold}-A-{target}.wav"
        sf.write(str(root / "audio" / name),
                 (0.1 * rng.standard_normal(8000)).astype(np.float32), 8000)
        rows.append({"filename": name, "fold": fold, "target": target,
                     "category": ["dog", "rain"][target], "esc10": True,
                     "src_file": 100000 + fold, "take": "A"})
    pd.DataFrame(rows).to_csv(root / "meta" / "esc50.csv", index=False)
    monkeypatch.setattr(extract, "ROOT", tmp_path)
    monkeypatch.setattr(cache_mod, "REPO_ROOT", tmp_path)
    monkeypatch.setitem(extract.CHECKPOINTS, "fake", None)
    monkeypatch.setattr(extract, "load_adapter", lambda name: FakeAdapter())
    monkeypatch.setitem(
        extract.CHECKPOINTS, "fake",
        type("C", (), {"checkpoint": "none", "revision": "r-test", "sha256": None})(),
    )
    return tmp_path


def test_extraction_alignment_and_cache(tiny_esc50, capsys):
    extract.extract_one_model("fake", "esc50", batch_size=2, device="cpu", smoke_n=None)
    X, filenames, fold_ids, label_ids, meta = load_embeddings("fake", "esc50")
    assert X.shape == (5, 4) and np.isfinite(X).all()
    assert list(fold_ids) == [1, 2, 3, 4, 5]          # official folds preserved
    assert list(label_ids) == [1, 0, 1, 0, 1]         # target column, meta order
    meta_df = pd.read_csv(
        tiny_esc50 / "data/raw/esc50/ESC-50-master/meta/esc50.csv")
    verify_alignment(filenames, meta_df, ESC50)       # row order == CSV order
    assert meta["revision"] == "r-test"
    assert X[0, 2] == 40000                            # 5 s @ 8 kHz after fix_duration

    # second run: metadata-keyed cache hit, no re-extraction
    extract.extract_one_model("fake", "esc50", batch_size=2, device="cpu", smoke_n=None)
    assert "cache is current, skipping" in capsys.readouterr().out


def test_extraction_rejects_nan(tiny_esc50):
    class NanAdapter(FakeAdapter):
        def embed_batch(self, wavs):
            out = super().embed_batch(wavs)
            out[0, 0] = np.nan
            return out

    extract.load_adapter = lambda name: NanAdapter()
    with pytest.raises(ValueError, match="NaN"):
        extract.extract_one_model("fake", "esc50", batch_size=2,
                                  device="cpu", smoke_n=None)


def test_smoke_writes_only_under_smoke_dir(tiny_esc50):
    extract.extract_one_model("fake", "esc50", batch_size=2, device="cpu", smoke_n=3)
    assert (tiny_esc50 / "data/smoke/embeddings/fake/esc50/embeddings.npz").exists()
    assert not (tiny_esc50 / "data/embeddings/fake/esc50/embeddings.npz").exists()
    X, filenames, *_ = load_embeddings("fake", "esc50", smoke=True)
    assert X.shape == (3, 4)
