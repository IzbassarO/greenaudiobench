"""Frozen-embedding linear probes (protocol P2). Implemented at M3.

Planned interface:
    run_probe(embeddings, meta, spec, seeds=(0, 1, 2, 3, 4)) -> pd.DataFrame
    - sklearn LogisticRegression (lbfgs, max_iter=2000)
    - C selected from {0.01, 0.1, 1, 10, 100} by inner CV on training folds
    - official outer folds via gab.folds.iter_official_folds
"""
