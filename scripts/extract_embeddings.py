#!/usr/bin/env python3
"""M2 — extract frozen embeddings for every P2 model on a dataset.

OFFICIAL runs (write to data/embeddings/): require CUDA (Colab T4), a clean
git tree (unless --allow-dirty), and fully pinned checkpoints. Embeddings are
always extracted in fp32 (fp16 is an efficiency-only ablation in M5).

SMOKE runs (--smoke N): first N clips, CPU allowed, write ONLY under
data/smoke/ — can never be mistaken for official caches.

Idempotent: a cache with matching checkpoint+preprocessing metadata is skipped.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gab.audio import DECODE_ID, fix_duration, load_audio  # noqa: E402
from gab.cache import cache_dir, is_cached, save_embeddings  # noqa: E402
from gab.datasets import SPECS, audio_path, dataset_root, load_meta, smoke_subset  # noqa: E402
from gab.models.registry import CHECKPOINTS, MODEL_ORDER, assert_pinned, load_adapter  # noqa: E402
from gab.utils import assert_official_run, run_metadata  # noqa: E402


def log(msg: str) -> None:
    print(f"[extract] {msg}", flush=True)


def extract_one_model(name: str, dataset: str, batch_size: int,
                      device: str, smoke_n: int | None) -> None:
    spec = SPECS[dataset]
    ckpt = CHECKPOINTS[name]
    smoke = smoke_n is not None
    meta = load_meta(spec, ROOT / "data")
    if smoke:
        # fold-balanced subset, NOT the first N rows: ESC-50 metadata is sorted
        # by filename (which starts with the fold id), so a head slice would
        # contain fold 1 only and M3's official-fold probe could not run
        meta = smoke_subset(meta, spec, smoke_n)
    root = dataset_root(spec, ROOT / "data")

    adapter = load_adapter(name)
    adapter.load(device=device, fp16=False)
    info = adapter.info

    expected_meta = {
        "model": name,
        "checkpoint": ckpt.checkpoint,
        "revision": ckpt.revision,
        "checkpoint_sha256": ckpt.sha256,
        "decode_id": DECODE_ID,
        "target_sample_rate": info.sample_rate,
        "duration_policy": info.duration_policy,
        "preprocess_id": info.preprocess_id,
        "embedding_layer": info.embedding_layer,
    }
    if is_cached(name, dataset, expected_meta, smoke):
        log(f"{name}/{dataset}: cache is current, skipping")
        return

    seconds = float(info.duration_policy.split(":")[1].rstrip("s"))
    filenames, fold_ids, label_ids, chunks = [], [], [], []
    batch: list[np.ndarray] = []
    t0 = time.monotonic()
    for i, (_, row) in enumerate(meta.iterrows()):
        wav = load_audio(audio_path(spec, root, row), info.sample_rate)
        batch.append(fix_duration(wav, info.sample_rate, seconds))
        filenames.append(row[spec.filename_column])
        fold_ids.append(int(row["fold"]))
        label_ids.append(int(row[spec.label_id_column]))
        if len(batch) == batch_size or i == len(meta) - 1:
            chunks.append(adapter.check_output(adapter.embed_batch(batch), len(batch)))
            batch = []
            if (i + 1) % (batch_size * 10) == 0:
                log(f"{name}/{dataset}: {i + 1}/{len(meta)} clips "
                    f"({time.monotonic() - t0:.0f}s)")

    X = np.concatenate(chunks, axis=0)
    if len(X) != len(meta):
        raise RuntimeError(f"{name}: extracted {len(X)} != {len(meta)} clips")
    save_embeddings(cache_dir(name, dataset, smoke), X, filenames,
                    np.array(fold_ids), np.array(label_ids), expected_meta)
    log(f"{name}/{dataset}: saved {X.shape} embeddings "
        f"({'SMOKE' if smoke else 'official'}) in {time.monotonic() - t0:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="esc50", choices=sorted(SPECS))
    parser.add_argument("--models", nargs="+", default=list(MODEL_ORDER),
                        choices=list(MODEL_ORDER))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--smoke", type=int, default=None, metavar="N",
                        help="smoke run on a deterministic fold-balanced subset of "
                             "at least N clips (covers every official fold); "
                             "CPU allowed, writes only to data/smoke/")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="debug override for the clean-tree provenance guard")
    args = parser.parse_args()

    models = [m for m in MODEL_ORDER if m in args.models]  # keep protocol order
    if args.smoke is None:
        assert_pinned(models)
        assert_official_run(require_cuda=True, allow_dirty=args.allow_dirty)
        device = "cuda"
    else:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"SMOKE MODE: {args.smoke} clips on {device}, output under data/smoke/")

    log(f"run metadata: {run_metadata()}")
    for name in models:
        extract_one_model(name, args.dataset, args.batch_size, device, args.smoke)
    log("done.")


if __name__ == "__main__":
    main()
