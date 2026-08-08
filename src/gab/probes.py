"""Frozen-embedding linear probes (protocol P2) — deterministic, fold-faithful.

Protocol (post-audit, C1/C2):
- probe = StandardScaler (fit on training folds only) + LogisticRegression
  (solver=lbfgs, max_iter=2000). lbfgs on fixed data is deterministic, so
  there is exactly ONE result per official outer fold — dispersion is
  reported across official folds, never described as seed variance.
- C is selected from {0.01, 0.1, 1, 10, 100} by leave-one-official-fold-out
  inner CV on the training folds (sklearn PredefinedSplit keyed by the
  ORIGINAL official fold ids — no random KFold, no clip-level reshuffling).
  Ties break toward the smallest C (stronger regularization), deterministically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import PredefinedSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

C_GRID: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
PROBE_ID = "standardized_logreg_lbfgs_maxiter2000"


def _make_probe(C: float):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, solver="lbfgs", max_iter=2000),
    )


def select_C_inner_cv(
    X_train: np.ndarray, y_train: np.ndarray, train_fold_ids: np.ndarray
) -> tuple[float, dict[float, float]]:
    """Pick C by rotating a validation fold over the OFFICIAL training folds.

    Each official training fold serves as the inner validation set exactly
    once (PredefinedSplit); the probe is fit on the remaining training folds.
    Returns (best_C, {C: mean_inner_accuracy}).
    """
    unique_folds = np.unique(train_fold_ids)
    if len(unique_folds) < 2:
        raise ValueError("inner CV needs at least 2 official training folds")
    split = PredefinedSplit(test_fold=train_fold_ids)
    scores: dict[float, float] = {}
    for C in C_GRID:
        accs = []
        for inner_train, inner_val in split.split():
            probe = _make_probe(C)
            probe.fit(X_train[inner_train], y_train[inner_train])
            accs.append(accuracy_score(y_train[inner_val], probe.predict(X_train[inner_val])))
        scores[C] = float(np.mean(accs))
    # deterministic argmax: highest accuracy, ties -> smallest C
    best_C = max(sorted(scores), key=lambda c: scores[c])
    return best_C, scores


def run_probe(
    X: np.ndarray,
    y: np.ndarray,
    fold_ids: np.ndarray,
    dataset: str,
    model: str,
) -> pd.DataFrame:
    """Full outer loop over official folds. One row per outer fold.

    X: (n_clips, dim) frozen embeddings; y: integer labels; fold_ids: the
    official fold id of every clip — all three aligned row-for-row.
    """
    if not (len(X) == len(y) == len(fold_ids)):
        raise ValueError("X, y and fold_ids must be aligned row-for-row")
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("embeddings contain NaN/Inf — refusing to fit probes")

    rows = []
    for outer_fold in sorted(np.unique(fold_ids)):
        test_mask = fold_ids == outer_fold
        X_train, y_train = X[~test_mask], y[~test_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        best_C, inner_scores = select_C_inner_cv(X_train, y_train, fold_ids[~test_mask])
        probe = _make_probe(best_C)
        probe.fit(X_train, y_train)
        pred = probe.predict(X_test)
        rows.append({
            "dataset": dataset,
            "model": model,
            "probe": PROBE_ID,
            "outer_fold": int(outer_fold),
            "selected_C": best_C,
            "inner_cv_scores": ";".join(f"{c}:{inner_scores[c]:.4f}" for c in C_GRID),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "accuracy": float(accuracy_score(y_test, pred)),
            "macro_f1": float(f1_score(y_test, pred, average="macro")),
        })
    return pd.DataFrame(rows)


def aggregate_across_folds(per_fold: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std ACROSS OFFICIAL OUTER FOLDS (this is fold dispersion,
    not seed variance) for each dataset × model."""
    return (
        per_fold.groupby(["dataset", "model"])
        .agg(
            n_folds=("outer_fold", "nunique"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
        )
        .reset_index()
    )
