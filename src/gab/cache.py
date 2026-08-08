"""Embedding cache: .npz arrays + metadata.json proving preprocessing identity.

Layout: data/embeddings/<model>/<dataset>/embeddings.npz + metadata.json
(smoke runs: data/smoke/embeddings/... — never mixes with official caches).

Alignment guarantee: arrays are stored in the EXACT row order of the official
metadata CSV, and the filenames array is stored alongside so any consumer can
re-verify alignment against the metadata instead of trusting positions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .datasets import DatasetSpec
from .utils import REPO_ROOT, run_metadata


def cache_dir(model: str, dataset: str, smoke: bool = False) -> Path:
    base = REPO_ROOT / "data" / ("smoke/embeddings" if smoke else "embeddings")
    return base / model / dataset


def save_embeddings(
    out_dir: Path,
    X: np.ndarray,
    filenames: list[str],
    fold_ids: np.ndarray,
    label_ids: np.ndarray,
    metadata: dict,
) -> None:
    if not (len(X) == len(filenames) == len(fold_ids) == len(label_ids)):
        raise ValueError("embeddings/filenames/folds/labels must be aligned")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "embeddings.npz",
        X=np.asarray(X, dtype=np.float32),
        filenames=np.asarray(filenames),
        fold_ids=np.asarray(fold_ids, dtype=np.int64),
        label_ids=np.asarray(label_ids, dtype=np.int64),
    )
    full_meta = {**metadata, "n_clips": int(len(X)), "embed_dim": int(X.shape[1]),
                 "run": run_metadata()}
    (out_dir / "metadata.json").write_text(json.dumps(full_meta, indent=2))


def load_embeddings(model: str, dataset: str, smoke: bool = False):
    """Returns (X, filenames, fold_ids, label_ids, metadata)."""
    d = cache_dir(model, dataset, smoke)
    npz = np.load(d / "embeddings.npz", allow_pickle=False)
    metadata = json.loads((d / "metadata.json").read_text())
    return npz["X"], npz["filenames"].tolist(), npz["fold_ids"], npz["label_ids"], metadata


def is_cached(model: str, dataset: str, expected: dict, smoke: bool = False) -> bool:
    """Cache hit only when metadata proves same checkpoint + preprocessing."""
    d = cache_dir(model, dataset, smoke)
    if not (d / "embeddings.npz").exists() or not (d / "metadata.json").exists():
        return False
    meta = json.loads((d / "metadata.json").read_text())
    return all(meta.get(k) == v for k, v in expected.items())


def verify_alignment(filenames: list[str], meta_df: pd.DataFrame, spec: DatasetSpec) -> None:
    """Cached row order must equal the official metadata CSV row order."""
    expected = meta_df[spec.filename_column].tolist()
    if filenames != expected:
        raise ValueError(
            f"embedding cache is NOT aligned with {spec.pretty_name} metadata "
            "(row order differs) — re-extract embeddings"
        )
