#!/usr/bin/env python3
"""M3 — deterministic frozen linear probes on cached embeddings.

Runs on CPU (sklearn); still an official stage: requires a clean git tree
(unless --allow-dirty) and refuses to run on smoke caches in official mode.
Appends one row per (model, outer fold) to results/accuracy.csv — append-only,
never rewrites existing rows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gab.cache import load_embeddings, verify_alignment  # noqa: E402
from gab.datasets import SPECS, load_meta  # noqa: E402
from gab.models.registry import MODEL_ORDER  # noqa: E402
from gab.probes import aggregate_across_folds, run_probe  # noqa: E402
from gab.utils import assert_official_run, run_metadata, smoke_output_dir  # noqa: E402


def log(msg: str) -> None:
    print(f"[probes] {msg}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="esc50", choices=sorted(SPECS))
    parser.add_argument("--models", nargs="+", default=list(MODEL_ORDER),
                        choices=list(MODEL_ORDER))
    parser.add_argument("--smoke", action="store_true",
                        help="use smoke caches; write to data/smoke/, not results/")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    if not args.smoke:
        # official but CPU-legitimate (sklearn): no CUDA requirement
        meta_run = assert_official_run(require_cuda=False, allow_dirty=args.allow_dirty)
    else:
        meta_run = run_metadata()
        log("SMOKE MODE: reading data/smoke/ caches, writing to data/smoke/")

    spec = SPECS[args.dataset]
    meta_df = load_meta(spec, ROOT / "data")

    frames = []
    for name in [m for m in MODEL_ORDER if m in args.models]:
        try:
            X, filenames, fold_ids, label_ids, cache_meta = load_embeddings(
                name, args.dataset, smoke=args.smoke)
        except FileNotFoundError:
            log(f"{name}/{args.dataset}: no embedding cache, skipping "
                f"(run extract_embeddings.py first)")
            continue
        if args.smoke:
            # the smoke cache holds a fold-balanced subset (gab.datasets.
            # smoke_subset), not a metadata prefix — align against exactly the
            # official rows it names, still in official metadata order
            sub = meta_df[meta_df[spec.filename_column].isin(filenames)]
        else:
            sub = meta_df
        verify_alignment(filenames, sub, spec)
        per_fold = run_probe(X, label_ids, fold_ids, args.dataset, name)
        for col, val in {**meta_run,
                         "embeddings_checkpoint_revision": cache_meta["revision"]}.items():
            per_fold[col] = val
        frames.append(per_fold)
        log(f"{name}: per-fold accuracy "
            + ", ".join(f"f{r.outer_fold}={r.accuracy:.4f}" for r in per_fold.itertuples()))

    if not frames:
        raise SystemExit("no models had embedding caches — nothing to do")
    results = pd.concat(frames, ignore_index=True)

    out = (smoke_output_dir() / "accuracy_smoke.csv") if args.smoke \
        else ROOT / "results" / "accuracy.csv"
    header = not out.exists()
    results.to_csv(out, mode="a", header=header, index=False)
    log(f"appended {len(results)} rows to {out}")

    agg = aggregate_across_folds(results)
    log("mean ± std ACROSS OFFICIAL FOLDS:\n" + agg.to_string(index=False))


if __name__ == "__main__":
    main()
