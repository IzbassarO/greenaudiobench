#!/usr/bin/env python3
"""M4 — protocol P1: zero-shot CLAP classification on official folds.

For each CLAP-family model (laion_clap, ms_clap) and each prompt template
(primary "a sound of {class_name}" plus the one planned ablation template),
all class prompts are embedded ONCE, each clip's audio embedding is compared
to them by cosine similarity (both sides explicitly L2-normalized), and the
prediction is the argmax class. Class names come from the official metadata
label column with underscores replaced by spaces.

Audio embeddings REUSE the P2 cache (data/embeddings/<model>/<dataset>) when
it exists and its metadata revision matches the registry pin — zero-shot CLAP
audio embeddings are identical to the P2 frozen embeddings. Otherwise they are
computed fresh through the SAME canonical path as extract_embeddings.py:
gab.audio.load_audio at the adapter's sample rate + fix_duration per the
adapter's duration_policy. The `audio_embedding_source` column records which
("cache" or "fresh").

Every row carries weak_zero_shot=True (CLAP pretraining corpora may overlap
ESC-50/US8K sources; required by CLAUDE.md, discussed honestly in the paper).

OFFICIAL runs append one row per model x template x official fold to
results/zeroshot.csv — require CUDA + clean git tree (unless --allow-dirty)
+ fully pinned checkpoints. SMOKE runs (--smoke N): first N clips, CPU
allowed, write ONLY data/smoke/zeroshot_smoke.csv — never results/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gab.audio import DECODE_ID, fix_duration, load_audio  # noqa: E402
from gab.cache import load_embeddings, verify_alignment  # noqa: E402
from gab.datasets import (  # noqa: E402
    SPECS, DatasetSpec, audio_path, dataset_root, load_meta, smoke_subset,
)
from gab.models.registry import CHECKPOINTS, assert_pinned, load_adapter  # noqa: E402
from gab.utils import assert_official_run, run_metadata, smoke_output_dir  # noqa: E402

#: P1 is CLAP-family only (the two models with a text tower), fixed order.
ZEROSHOT_MODELS = ("laion_clap", "ms_clap")

#: Prompt templates: primary + the ONE planned ablation (CLAUDE.md P1).
TEMPLATES = {
    "primary": "a sound of {class_name}",
    "alternative": "this is the sound of {class_name}",
}


def log(msg: str) -> None:
    print(f"[zeroshot] {msg}", flush=True)


def class_names_by_id(meta_df: pd.DataFrame, spec: DatasetSpec) -> list[str]:
    """Human-readable class names ordered by integer label id 0..K-1.

    Underscores in the official category names become spaces, so the prompt
    for ESC-50 "chirping_birds" reads "a sound of chirping birds". The
    returned order guarantees argmax index == official label id.
    """
    pairs = (
        meta_df[[spec.label_id_column, spec.label_column]]
        .drop_duplicates()
        .sort_values(spec.label_id_column)
    )
    ids = pairs[spec.label_id_column].tolist()
    if ids != list(range(len(ids))):
        raise ValueError(f"{spec.key}: label ids are not contiguous 0..K-1: {ids[:10]}")
    return [str(c).replace("_", " ") for c in pairs[spec.label_column]]


def build_prompts(template: str, class_names: list[str]) -> list[str]:
    return [template.format(class_name=name) for name in class_names]


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Explicit row-wise L2 normalization; refuses zero-norm rows."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"expected 2D embeddings, got shape {x.shape}")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.all(norms > 0):
        raise ValueError("zero-norm embedding cannot be cosine-normalized")
    return x / norms


def cosine_scores(audio_emb: np.ndarray, text_emb: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix (n_clips, n_classes).

    BOTH sides are explicitly L2-normalized here — never rely on a model
    having normalized its projection output.
    """
    A = l2_normalize(audio_emb)
    T = l2_normalize(text_emb)
    if A.shape[1] != T.shape[1]:
        raise ValueError(
            f"audio embed dim {A.shape[1]} != text embed dim {T.shape[1]}"
        )
    return A @ T.T


def embed_texts_checked(adapter, prompts: list[str]) -> np.ndarray:
    """Call adapter.embed_texts and validate shape/finiteness."""
    if not hasattr(adapter, "embed_texts"):
        raise TypeError(
            f"{adapter.info.name}: adapter has no embed_texts() — "
            "P1 zero-shot needs the CLAP text tower"
        )
    T = np.asarray(adapter.embed_texts(prompts), dtype=np.float32)
    if T.ndim != 2 or T.shape[0] != len(prompts):
        raise ValueError(
            f"{adapter.info.name}: text embeddings shape {T.shape} != ({len(prompts)}, dim)"
        )
    if not np.isfinite(T).all():
        raise ValueError(f"{adapter.info.name}: NaN/Inf in text embeddings")
    return T


def per_fold_metrics(
    preds: np.ndarray, label_ids: np.ndarray, fold_ids: np.ndarray, n_classes: int
) -> pd.DataFrame:
    """One row per OFFICIAL fold present: accuracy, macro-F1 (sklearn), n_clips."""
    from sklearn.metrics import accuracy_score, f1_score

    preds = np.asarray(preds)
    label_ids = np.asarray(label_ids)
    fold_ids = np.asarray(fold_ids)
    if not (len(preds) == len(label_ids) == len(fold_ids)):
        raise ValueError("preds/labels/folds must be aligned")
    rows = []
    for fold in sorted(np.unique(fold_ids).tolist()):
        m = fold_ids == fold
        rows.append({
            "fold": int(fold),
            "n_clips": int(m.sum()),
            "accuracy": float(accuracy_score(label_ids[m], preds[m])),
            "macro_f1": float(f1_score(
                label_ids[m], preds[m], average="macro",
                labels=list(range(n_classes)), zero_division=0,
            )),
        })
    return pd.DataFrame(rows)


def cached_audio_embeddings(
    name: str, spec: DatasetSpec, meta_df: pd.DataFrame, smoke: bool
):
    """P2 embedding cache for this model, or None when absent/stale.

    A cache is usable only when its recorded checkpoint revision matches the
    registry pin AND its decode identity matches the current canonical decode
    AND its row order aligns with the official metadata (verify_alignment).
    Smoke mode may use a prefix of a larger smoke cache.
    """
    try:
        X, filenames, fold_ids, label_ids, cache_meta = load_embeddings(
            name, spec.key, smoke=smoke)
    except FileNotFoundError:
        return None
    ckpt = CHECKPOINTS[name]
    if cache_meta.get("revision") != ckpt.revision:
        log(f"{name}/{spec.key}: cache revision {cache_meta.get('revision')!r} != "
            f"registry pin {ckpt.revision!r} — computing audio embeddings fresh")
        return None
    if cache_meta.get("decode_id", DECODE_ID) != DECODE_ID:
        log(f"{name}/{spec.key}: cache decode_id is stale — computing fresh")
        return None
    n = len(meta_df)
    if len(filenames) != n:
        log(f"{name}/{spec.key}: cache has {len(filenames)} rows, need {n} — "
            "computing fresh")
        return None
    verify_alignment(filenames, meta_df, spec)
    return X, np.asarray(fold_ids), np.asarray(label_ids)


def fresh_audio_embeddings(
    adapter, meta_df: pd.DataFrame, spec: DatasetSpec, data_root: Path, batch_size: int
):
    """Same canonical decode + duration path as extract_embeddings.py."""
    info = adapter.info
    seconds = float(info.duration_policy.split(":")[1].rstrip("s"))
    fold_ids: list[int] = []
    label_ids: list[int] = []
    chunks: list[np.ndarray] = []
    batch: list[np.ndarray] = []
    for i, (_, row) in enumerate(meta_df.iterrows()):
        wav = load_audio(audio_path(spec, data_root, row), info.sample_rate)
        batch.append(fix_duration(wav, info.sample_rate, seconds))
        fold_ids.append(int(row["fold"]))
        label_ids.append(int(row[spec.label_id_column]))
        if len(batch) == batch_size or i == len(meta_df) - 1:
            chunks.append(adapter.check_output(adapter.embed_batch(batch), len(batch)))
            batch = []
            if (i + 1) % (batch_size * 10) == 0:
                log(f"{info.name}/{spec.key}: embedded {i + 1}/{len(meta_df)} clips")
    X = np.concatenate(chunks, axis=0)
    if len(X) != len(meta_df):
        raise RuntimeError(f"{info.name}: embedded {len(X)} != {len(meta_df)} clips")
    return X, np.asarray(fold_ids), np.asarray(label_ids)


def evaluate_model(
    name: str,
    adapter,
    meta_df: pd.DataFrame,
    class_names: list[str],
    spec: DatasetSpec,
    data_root: Path,
    batch_size: int,
    smoke: bool,
) -> pd.DataFrame:
    """All templates for one model: one output row per template x official fold."""
    cached = cached_audio_embeddings(name, spec, meta_df, smoke)
    if cached is not None:
        A, fold_ids, label_ids = cached
        source = "cache"
        log(f"{name}/{spec.key}: audio embeddings reused from P2 cache "
            f"({A.shape[0]} clips)")
    else:
        A, fold_ids, label_ids = fresh_audio_embeddings(
            adapter, meta_df, spec, data_root, batch_size)
        source = "fresh"
        log(f"{name}/{spec.key}: audio embeddings computed fresh ({A.shape[0]} clips)")

    frames = []
    for template_id, template in TEMPLATES.items():
        prompts = build_prompts(template, class_names)
        T = embed_texts_checked(adapter, prompts)
        preds = cosine_scores(A, T).argmax(axis=1)
        pf = per_fold_metrics(preds, label_ids, fold_ids, len(class_names))
        pf.insert(0, "dataset", spec.key)
        pf.insert(1, "model", name)
        pf.insert(2, "template_id", template_id)
        pf.insert(3, "prompt_template", template)
        pf["weak_zero_shot"] = True
        pf["audio_embedding_source"] = source
        pf["checkpoint_revision"] = CHECKPOINTS[name].revision
        frames.append(pf)
        log(f"{name} [{template_id}]: per-fold accuracy "
            + ", ".join(f"f{r.fold}={r.accuracy:.4f}" for r in pf.itertuples()))
    return pd.concat(frames, ignore_index=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="esc50", choices=sorted(SPECS))
    parser.add_argument("--models", nargs="+", default=list(ZEROSHOT_MODELS),
                        choices=list(ZEROSHOT_MODELS))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--smoke", type=int, default=None, metavar="N",
                        help="smoke run on a deterministic fold-balanced subset "
                             "of at least N clips (same subset as "
                             "extract_embeddings.py --smoke N); writes ONLY "
                             "data/smoke/zeroshot_smoke.csv")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="debug override for the clean-tree provenance guard")
    args = parser.parse_args(argv)

    models = [m for m in ZEROSHOT_MODELS if m in args.models]  # keep P1 order
    smoke = args.smoke is not None
    if not smoke:
        assert_pinned(models)
        meta_run = assert_official_run(require_cuda=True, allow_dirty=args.allow_dirty)
        device = "cuda"
    else:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        meta_run = run_metadata()
        log(f"SMOKE MODE: {args.smoke} clips on {device}, "
            "output ONLY under data/smoke/")

    log(f"run metadata: {meta_run}")
    spec = SPECS[args.dataset]
    full_meta = load_meta(spec, ROOT / "data")
    # Prompts always cover ALL official classes (built from the FULL metadata,
    # before any smoke slicing) so argmax index == official label id.
    class_names = class_names_by_id(full_meta, spec)
    if not smoke and len(class_names) != spec.expected_classes:
        raise RuntimeError(
            f"{spec.key}: {len(class_names)} classes != official "
            f"{spec.expected_classes}")
    # smoke uses the SAME fold-balanced subset as extract_embeddings.py, so the
    # P2 smoke caches stay reusable here (never a metadata prefix)
    meta_df = smoke_subset(full_meta, spec, args.smoke) if smoke else full_meta
    data_root = dataset_root(spec, ROOT / "data")

    frames = []
    for name in models:
        adapter = load_adapter(name)
        adapter.load(device=device, fp16=False)  # P1 is always fp32
        res = evaluate_model(name, adapter, meta_df, class_names, spec,
                             data_root, args.batch_size, smoke)
        for col, val in meta_run.items():
            res[col] = val
        frames.append(res)

    results = pd.concat(frames, ignore_index=True)
    out = (smoke_output_dir() / "zeroshot_smoke.csv") if smoke \
        else ROOT / "results" / "zeroshot.csv"
    header = not out.exists()
    results.to_csv(out, mode="a", header=header, index=False)
    log(f"appended {len(results)} rows to {out}")

    agg = (results.groupby(["model", "template_id"])["accuracy"]
           .agg(["mean", "std"]).reset_index())
    log("accuracy ACROSS OFFICIAL FOLDS (mean, std):\n" + agg.to_string(index=False))


if __name__ == "__main__":
    main()
