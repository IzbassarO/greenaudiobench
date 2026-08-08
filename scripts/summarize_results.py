#!/usr/bin/env python3
"""M7 — validate the official ESC-50 result CSVs and derive summary tables.

Inputs (immutable experimental evidence, never modified):
    results/accuracy.csv     M3 frozen linear probes
    results/zeroshot.csv     M4 CLAP zero-shot
    results/efficiency.csv   M5 latency + NVML energy

Outputs:
    reports/M7_RESULTS_VALIDATION.md
    results/tables/*.csv, results/tables/*.md

Every number is recomputed from the CSVs — nothing is hardcoded. Failed or
unsupported configurations are carried through explicitly and never dropped.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
REPORTS = ROOT / "reports"

#: Canonical display names used in every table and figure.
DISPLAY = {
    "ast": "AST",
    "laion_clap": "LAION-CLAP",
    "panns_cnn14": "PANNs CNN14",
    "beats": "BEATs",
    "ms_clap": "MS-CLAP",
}
MODEL_ORDER = ["ast", "laion_clap", "panns_cnn14", "beats", "ms_clap"]
ZS_MODELS = ["laion_clap", "ms_clap"]
OFFICIAL_FOLDS = [1, 2, 3, 4, 5]

EXPECTED_COMMIT_PROBES = "d20e86fe2617d8904ee95010904cdbdf8cfb12eb"
EXPECTED_COMMIT_EFFICIENCY = "93eaacb9390a0eb42e3266021e0c905a04f2fc02"
EXPECTED_GPU = "Tesla T4"


class Checks:
    """Collects pass/fail checks for the validation report."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, group: str, name: str, expected, observed) -> None:
        ok = expected == observed
        self.rows.append({"group": group, "check": name, "expected": expected,
                          "observed": observed, "status": "PASS" if ok else "FAIL"})

    def note(self, group: str, name: str, observed) -> None:
        """Recorded fact, not a pass/fail assertion."""
        self.rows.append({"group": group, "check": name, "expected": "(informational)",
                          "observed": observed, "status": "INFO"})

    @property
    def failures(self) -> list[dict]:
        return [r for r in self.rows if r["status"] == "FAIL"]


def validate_accuracy(df: pd.DataFrame, c: Checks) -> None:
    g = "accuracy.csv (M3)"
    c.add(g, "row count", 25, len(df))
    c.add(g, "datasets", ["esc50"], sorted(df["dataset"].unique()))
    c.add(g, "models", sorted(MODEL_ORDER), sorted(df["model"].unique()))
    c.add(g, "official folds", OFFICIAL_FOLDS, sorted(df["outer_fold"].unique()))
    c.add(g, "rows per model", {m: 5 for m in sorted(MODEL_ORDER)},
          df.groupby("model").size().to_dict())
    c.add(g, "n_train (all rows)", [1600], sorted(df["n_train"].unique()))
    c.add(g, "n_test (all rows)", [400], sorted(df["n_test"].unique()))
    c.add(g, "n_train + n_test == 2000 clips", True,
          bool(((df["n_train"] + df["n_test"]) == 2000).all()))
    c.add(g, "git_dirty", [False], sorted(df["git_dirty"].unique()))
    c.add(g, "git_commit", [EXPECTED_COMMIT_PROBES], sorted(df["git_commit"].unique()))
    c.add(g, "gpu_name", [EXPECTED_GPU], sorted(df["gpu_name"].unique()))
    c.add(g, "missing accuracy values", 0, int(df["accuracy"].isna().sum()))
    c.add(g, "missing macro_f1 values", 0, int(df["macro_f1"].isna().sum()))
    c.add(g, "one row per model x fold (no duplicates)", 25,
          len(df.drop_duplicates(["model", "outer_fold"])))
    c.add(g, "probe identifier unique", 1, df["probe"].nunique())
    c.note(g, "selected C values observed", sorted(df["selected_C"].unique()))
    c.note(g, "checkpoint revisions", df.groupby("model")["embeddings_checkpoint_revision"]
           .first().to_dict())


def validate_zeroshot(df: pd.DataFrame, c: Checks) -> None:
    g = "zeroshot.csv (M4)"
    c.add(g, "row count", 20, len(df))
    c.add(g, "datasets", ["esc50"], sorted(df["dataset"].unique()))
    c.add(g, "models (CLAP family only)", sorted(ZS_MODELS), sorted(df["model"].unique()))
    c.add(g, "templates per model", 2, df.groupby("model")["template_id"].nunique().max())
    c.add(g, "official folds", OFFICIAL_FOLDS, sorted(df["fold"].unique()))
    c.add(g, "rows per model x template", {(m, t): 5 for m in sorted(ZS_MODELS)
                                           for t in sorted(df["template_id"].unique())},
          df.groupby(["model", "template_id"]).size().to_dict())
    c.add(g, "n_clips per fold", [400], sorted(df["n_clips"].unique()))
    c.add(g, "weak_zero_shot flagged", [True], sorted(df["weak_zero_shot"].unique()))
    c.add(g, "git_dirty", [False], sorted(df["git_dirty"].unique()))
    c.add(g, "git_commit", [EXPECTED_COMMIT_PROBES], sorted(df["git_commit"].unique()))
    c.add(g, "gpu_name", [EXPECTED_GPU], sorted(df["gpu_name"].unique()))
    c.add(g, "missing accuracy values", 0, int(df["accuracy"].isna().sum()))
    c.note(g, "prompt templates", sorted(df["prompt_template"].unique()))
    c.note(g, "audio embedding source", sorted(df["audio_embedding_source"].unique()))


def validate_efficiency(df: pd.DataFrame, c: Checks) -> None:
    g = "efficiency.csv (M5)"
    c.add(g, "row count", 20, len(df))
    c.add(g, "models", sorted(MODEL_ORDER), sorted(df["model"].unique()))
    c.add(g, "dtypes", ["fp16", "fp32"], sorted(df["dtype"].unique()))
    c.add(g, "batch sizes", [1, 32], sorted(df["batch_size"].unique()))
    c.add(g, "one row per model x dtype x batch", 20,
          len(df.drop_duplicates(["model", "dtype", "batch_size"])))
    c.add(g, "smoke rows present", [False], sorted(df["smoke"].unique()))
    c.add(g, "git_dirty", [False], sorted(df["git_dirty"].unique()))
    c.add(g, "git_commit", [EXPECTED_COMMIT_EFFICIENCY], sorted(df["git_commit"].unique()))
    c.add(g, "gpu_name", [EXPECTED_GPU], sorted(df["gpu_name"].unique()))

    ok = df[df["status"] == "ok"]
    c.add(g, "all fp32 configurations measured", 10, len(ok[ok["dtype"] == "fp32"]))
    c.add(g, "measured rows have latency", 0,
          int(ok["latency_ms_per_clip_median"].isna().sum()))
    c.add(g, "measured rows have total J/clip", 0, int(ok["j_per_clip_total"].isna().sum()))
    c.add(g, "measured rows have dynamic J/clip", 0,
          int(ok["j_per_clip_dynamic"].isna().sum()))
    c.add(g, "latency protocol: 100 timed x 3 repeats", [(300.0, 3.0)],
          sorted(set(zip(ok["n_timed_iterations"], ok["n_repeats"]))))
    c.add(g, "energy window >= 30 s everywhere", True,
          bool((ok["energy_window_s"] >= 30.0).all()))
    c.add(g, "clip divisor == batches x batch_size", True,
          bool(((ok["clips_processed"] % ok["batch_size"]) == 0).all()))
    c.add(g, "dynamic J/clip < total J/clip (idle subtracted)", True,
          bool((ok["j_per_clip_dynamic"] < ok["j_per_clip_total"]).all()))
    c.add(g, "single idle baseline across the run", 1, ok["idle_power_w"].nunique())

    failed = df[df["status"] != "ok"]
    c.note(g, "non-ok configurations (kept, never dropped)",
           {f"{r.model}/{r.dtype}/bs{r.batch_size}": f"{r.status}: {r.note}"
            for r in failed.itertuples()})
    c.note(g, "idle baseline (W)", round(float(ok["idle_power_w"].iloc[0]), 4))
    c.note(g, "GPU temp range before/after (C)",
           (float(ok["gpu_temp_before_c"].min()), float(ok["gpu_temp_after_c"].max())))
    c.note(g, "GMACs unavailable (fvcore failed)",
           sorted({f"{r.model}/{r.dtype}" for r in df.itertuples()
                   if pd.isna(r.gmacs_fvcore_b1)}))


def fmt(x, nd=4):
    return "" if pd.isna(x) else f"{float(x):.{nd}f}"


def write_md_table(path: Path, df: pd.DataFrame, title: str, note: str) -> None:
    lines = [f"# {title}", "", note, "",
             "| " + " | ".join(df.columns) + " |",
             "|" + "|".join("---" for _ in df.columns) + "|"]
    for row in df.itertuples(index=False):
        cells = ["" if pd.isna(v) else (f"{v:.4f}" if isinstance(v, float) else str(v))
                 for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")


def probe_summary(acc: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- std ACROSS THE FIVE OFFICIAL FOLDS (fold dispersion)."""
    g = (acc.groupby("model")
         .agg(n_folds=("outer_fold", "nunique"),
              accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
              macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"))
         .reset_index())
    g["model_display"] = g["model"].map(DISPLAY)
    g = g.sort_values("accuracy_mean", ascending=False).reset_index(drop=True)
    return g[["model", "model_display", "n_folds", "accuracy_mean", "accuracy_std",
              "macro_f1_mean", "macro_f1_std"]]


def zeroshot_summary(zs: pd.DataFrame) -> pd.DataFrame:
    g = (zs.groupby(["model", "template_id", "prompt_template"])
         .agg(n_folds=("fold", "nunique"),
              accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
              macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"))
         .reset_index())
    g["model_display"] = g["model"].map(DISPLAY)
    g["weak_zero_shot"] = True
    return g.sort_values(["model", "template_id"]).reset_index(drop=True)


def efficiency_summary(eff: pd.DataFrame) -> pd.DataFrame:
    cols = ["model", "dtype", "batch_size", "status", "note", "params_millions",
            "gmacs_fvcore_b1", "latency_ms_per_clip_median", "latency_ms_per_clip_p25",
            "latency_ms_per_clip_p75", "j_per_clip_total", "j_per_clip_dynamic",
            "idle_power_w", "clips_processed", "energy_window_s", "n_power_samples",
            "gpu_temp_before_c", "gpu_temp_after_c"]
    out = eff[cols].copy()
    out.insert(1, "model_display", out["model"].map(DISPLAY))
    out["model"] = pd.Categorical(out["model"], MODEL_ORDER, ordered=True)
    return out.sort_values(["model", "dtype", "batch_size"]).reset_index(drop=True)


def main_comparison(probe: pd.DataFrame, eff: pd.DataFrame) -> pd.DataFrame:
    """fp32 is the fair universal comparison: all five models measured there."""
    rows = []
    for _, p in probe.iterrows():
        m = p["model"]
        row = {
            "model": m,
            "model_display": DISPLAY[m],
            "accuracy_pct": 100 * p["accuracy_mean"],
            "accuracy_std_pct": 100 * p["accuracy_std"],
            "macro_f1_pct": 100 * p["macro_f1_mean"],
            "macro_f1_std_pct": 100 * p["macro_f1_std"],
            "params_millions": float(eff[eff["model"] == m]["params_millions"].iloc[0]),
        }
        for bs in (1, 32):
            sel = eff[(eff["model"] == m) & (eff["dtype"] == "fp32")
                      & (eff["batch_size"] == bs) & (eff["status"] == "ok")]
            row[f"fp32_bs{bs}_latency_ms_per_clip_median"] = (
                float(sel["latency_ms_per_clip_median"].iloc[0]) if len(sel) else float("nan"))
            row[f"fp32_bs{bs}_j_per_clip_dynamic"] = (
                float(sel["j_per_clip_dynamic"].iloc[0]) if len(sel) else float("nan"))
            row[f"fp32_bs{bs}_j_per_clip_total"] = (
                float(sel["j_per_clip_total"].iloc[0]) if len(sel) else float("nan"))
        fp16 = eff[(eff["model"] == m) & (eff["dtype"] == "fp16")]
        statuses = set(fp16["status"])
        row["fp16_status"] = "measured" if statuses == {"ok"} else (
            f"failed: {fp16['note'].dropna().iloc[0]}" if len(fp16["note"].dropna())
            else "failed")
        rows.append(row)
    return pd.DataFrame(rows)


def pareto_front(df: pd.DataFrame, benefit: str, cost: str) -> pd.Series:
    """Non-dominated set: dominated iff another row has >= benefit AND <= cost,
    with at least one strict inequality."""
    keep = []
    for i, a in df.iterrows():
        dominated = False
        for j, b in df.iterrows():
            if i == j:
                continue
            if (b[benefit] >= a[benefit] and b[cost] <= a[cost]
                    and (b[benefit] > a[benefit] or b[cost] < a[cost])):
                dominated = True
                break
        keep.append(not dominated)
    return pd.Series(keep, index=df.index)


def pareto_table(probe: pd.DataFrame, eff: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("A", "accuracy vs latency, fp32 batch 1", 1, "latency_ms_per_clip_median"),
        ("B", "accuracy vs dynamic J/clip, fp32 batch 1", 1, "j_per_clip_dynamic"),
        ("C", "accuracy vs latency, fp32 batch 32", 32, "latency_ms_per_clip_median"),
        ("D", "accuracy vs dynamic J/clip, fp32 batch 32", 32, "j_per_clip_dynamic"),
    ]
    acc = probe.set_index("model")["accuracy_mean"]
    frames = []
    for key, label, bs, cost in comparisons:
        sel = eff[(eff["dtype"] == "fp32") & (eff["batch_size"] == bs)
                  & (eff["status"] == "ok")].copy()
        sel["accuracy_mean"] = sel["model"].map(acc)
        sel = sel.dropna(subset=["accuracy_mean", cost]).reset_index(drop=True)
        sel["pareto_optimal"] = pareto_front(sel, "accuracy_mean", cost)
        out = sel[["model", "dtype", "batch_size", "accuracy_mean", cost,
                   "pareto_optimal"]].rename(columns={cost: "cost_value"})
        out.insert(0, "comparison", key)
        out.insert(1, "comparison_label", label)
        out.insert(2, "cost_metric", cost)
        out.insert(4, "model_display", out["model"].map(DISPLAY))
        frames.append(out.sort_values("cost_value").reset_index(drop=True))
    return pd.concat(frames, ignore_index=True)


def write_validation_report(checks: Checks, inputs: dict[str, pd.DataFrame]) -> None:
    REPORTS.mkdir(exist_ok=True)
    lines = [
        "# M7 — validation of the official ESC-50 result CSVs", "",
        "Generated by `scripts/summarize_results.py` from the immutable result",
        "files; no CSV content was modified. Checks marked INFO are recorded",
        "facts, not assertions.", "",
        "## Inputs", "",
        "| file | rows | columns | sha-verified provenance commit |",
        "|---|---|---|---|",
    ]
    for name, df in inputs.items():
        lines.append(f"| results/{name} | {len(df)} | {len(df.columns)} | "
                     f"{', '.join(sorted(df['git_commit'].unique()))} |")
    n_fail = len(checks.failures)
    lines += ["", f"## Verdict: {'PASS' if n_fail == 0 else f'FAIL ({n_fail} checks)'}",
              "", f"{len(checks.rows)} checks recorded, "
              f"{sum(1 for r in checks.rows if r['status'] == 'PASS')} PASS, "
              f"{sum(1 for r in checks.rows if r['status'] == 'INFO')} INFO, "
              f"{n_fail} FAIL.", "",
              "## Checks", "", "| group | check | expected | observed | status |",
              "|---|---|---|---|---|"]
    for r in checks.rows:
        exp = str(r["expected"]).replace("|", "\\|")
        obs = str(r["observed"]).replace("|", "\\|")
        lines.append(f"| {r['group']} | {r['check']} | `{exp}` | `{obs}` | {r['status']} |")
    (REPORTS / "M7_RESULTS_VALIDATION.md").write_text("\n".join(lines) + "\n")


def write_findings_report(probe, zsum, eff, main_tbl, par, acc, zs) -> None:
    """Factual evidence report — every number recomputed from the CSVs."""
    REPORTS.mkdir(exist_ok=True)
    L: list[str] = []
    p = probe.set_index("model")

    def pct(x):
        return f"{100 * x:.2f}"

    L += ["# M7 — ESC-50 factual findings (evidence report, not manuscript prose)", "",
          "All values computed by `scripts/summarize_results.py` from "
          "`results/accuracy.csv`, `results/zeroshot.csv` and "
          "`results/efficiency.csv`. Hardware: NVIDIA Tesla T4 (Google Colab), "
          "driver "
          f"{eff[eff['status'] == 'ok']['driver_version'].iloc[0]}, "
          f"CUDA {eff[eff['status'] == 'ok']['cuda_version'].iloc[0]}, "
          f"torch {eff[eff['status'] == 'ok']['torch_version'].iloc[0]}.", "",
          "## A. Frozen linear-probe ranking (protocol P2, fp32 embeddings)", "",
          "Accuracy and macro-F1 are means across the five official ESC-50 folds; "
          "the +/- value is the standard deviation across those folds (fold "
          "dispersion, not seed variance — the probe is deterministic).", "",
          "| rank | model | accuracy (%) | macro-F1 (%) | selected C per fold |",
          "|---|---|---|---|---|"]
    for i, r in enumerate(probe.itertuples(), 1):
        cs = acc[acc["model"] == r.model].sort_values("outer_fold")["selected_C"].tolist()
        L.append(f"| {i} | {DISPLAY[r.model]} | {pct(r.accuracy_mean)} +/- "
                 f"{pct(r.accuracy_std)} | {pct(r.macro_f1_mean)} +/- "
                 f"{pct(r.macro_f1_std)} | {cs} |")

    best, worst = probe.iloc[0], probe.iloc[-1]
    spread = 100 * (best["accuracy_mean"] - worst["accuracy_mean"])
    L += ["", f"Spread between best and worst model: **{spread:.2f} percentage "
          f"points** ({DISPLAY[best['model']]} {pct(best['accuracy_mean'])}% vs "
          f"{DISPLAY[worst['model']]} {pct(worst['accuracy_mean'])}%).", ""]

    L += ["## B. Zero-shot ranking (protocol P1, CLAP family only)", "",
          "All rows carry `weak_zero_shot=True`: the CLAP pretraining corpora may "
          "overlap the ESC-50 source recordings, so these are not clean zero-shot "
          "numbers.", "",
          "| model | template | prompt | accuracy (%) | macro-F1 (%) |",
          "|---|---|---|---|---|"]
    for r in zsum.sort_values("accuracy_mean", ascending=False).itertuples():
        L.append(f"| {DISPLAY[r.model]} | {r.template_id} | `{r.prompt_template}` | "
                 f"{pct(r.accuracy_mean)} +/- {pct(r.accuracy_std)} | "
                 f"{pct(r.macro_f1_mean)} +/- {pct(r.macro_f1_std)} |")

    L += ["", "### Template sensitivity and probe gap", ""]
    for m in ZS_MODELS:
        sub = zsum[zsum["model"] == m].sort_values("accuracy_mean", ascending=False)
        top, low = sub.iloc[0], sub.iloc[-1]
        sens = 100 * (top["accuracy_mean"] - low["accuracy_mean"])
        gap = 100 * (p.loc[m, "accuracy_mean"] - top["accuracy_mean"])
        L.append(f"- **{DISPLAY[m]}**: best template `{top['template_id']}` at "
                 f"{pct(top['accuracy_mean'])}%, worst `{low['template_id']}` at "
                 f"{pct(low['accuracy_mean'])}% -> absolute template sensitivity "
                 f"**{sens:.2f} pp**. Frozen probe ({pct(p.loc[m, 'accuracy_mean'])}%) "
                 f"exceeds the best zero-shot template by **{gap:.2f} pp**.")

    L += ["", "## C. Efficiency ranking (fp32, measured waveform -> embedding)", ""]
    for bs in (1, 32):
        sel = eff[(eff["dtype"] == "fp32") & (eff["batch_size"] == bs)
                  & (eff["status"] == "ok")].copy()
        sel = sel.sort_values("latency_ms_per_clip_median")
        L += [f"### fp32, batch size {bs}", "",
              "| rank by latency | model | median ms/clip | IQR (p25-p75) | "
              "dynamic J/clip | total J/clip | params (M) | GMACs (batch 1) |",
              "|---|---|---|---|---|---|---|---|"]
        for i, r in enumerate(sel.itertuples(), 1):
            gm = "n/a (fvcore failed)" if pd.isna(r.gmacs_fvcore_b1) else f"{r.gmacs_fvcore_b1:.3f}"
            L.append(f"| {i} | {DISPLAY[r.model]} | "
                     f"{r.latency_ms_per_clip_median:.2f} | "
                     f"{r.latency_ms_per_clip_p25:.2f}-{r.latency_ms_per_clip_p75:.2f} | "
                     f"{r.j_per_clip_dynamic:.4f} | {r.j_per_clip_total:.4f} | "
                     f"{r.params_millions:.1f} | {gm} |")
        L.append("")
        e_sorted = sel.sort_values("j_per_clip_dynamic")
        L.append("Ranked by dynamic energy: " + " < ".join(
            f"{DISPLAY[r.model]} ({r.j_per_clip_dynamic:.4f} J)"
            for r in e_sorted.itertuples()))
        L.append("")

    idle = float(eff[eff["status"] == "ok"]["idle_power_w"].iloc[0])
    L += [f"Idle GPU baseline subtracted from every dynamic value: **{idle:.3f} W** "
          "(measured over a >=30 s idle window before the measurement loops).", ""]

    L += ["## D. Pareto-optimal configurations (exact, computed programmatically)", ""]
    for key, group in par.groupby("comparison"):
        label = group["comparison_label"].iloc[0]
        front = group[group["pareto_optimal"]].sort_values("cost_value")
        dominated = group[~group["pareto_optimal"]]["model_display"].tolist()
        members = ", ".join(f"{r.model_display} (accuracy {pct(r.accuracy_mean)}%, "
                            f"cost {r.cost_value:.4f})" for r in front.itertuples())
        L.append(f"- **{key}. {label}** -> frontier: {members}. "
                 f"Dominated: {', '.join(dominated) if dominated else 'none'}.")

    L += ["", "## E. Relative comparisons (all computed from the CSVs)", ""]
    e1 = eff[(eff["dtype"] == "fp32") & (eff["batch_size"] == 1)
             & (eff["status"] == "ok")].set_index("model")
    e32 = eff[(eff["dtype"] == "fp32") & (eff["batch_size"] == 32)
              & (eff["status"] == "ok")].set_index("model")
    ms, ast_ = "ms_clap", "ast"
    L += [
        f"- At fp32 batch 1, {DISPLAY[ms]} is "
        f"**{e1.loc[ast_, 'latency_ms_per_clip_median'] / e1.loc[ms, 'latency_ms_per_clip_median']:.2f}x "
        f"faster** than {DISPLAY[ast_]} "
        f"({e1.loc[ms, 'latency_ms_per_clip_median']:.2f} vs "
        f"{e1.loc[ast_, 'latency_ms_per_clip_median']:.2f} ms/clip) and uses "
        f"**{100 * (1 - e1.loc[ms, 'j_per_clip_dynamic'] / e1.loc[ast_, 'j_per_clip_dynamic']):.1f}% "
        f"less dynamic energy per clip**, while scoring "
        f"**{100 * (p.loc[ms, 'accuracy_mean'] - p.loc[ast_, 'accuracy_mean']):.2f} pp higher** "
        "frozen-probe accuracy.",
        f"- {DISPLAY['panns_cnn14']} is the cheapest model at fp32 batch 1 "
        f"({e1.loc['panns_cnn14', 'j_per_clip_dynamic']:.4f} J/clip), but it buys "
        f"only "
        f"{100 * abs(1 - e1.loc['panns_cnn14', 'j_per_clip_dynamic'] / e1.loc[ms, 'j_per_clip_dynamic']):.1f}% "
        f"less dynamic energy than {DISPLAY[ms]} while scoring "
        f"**{100 * abs(p.loc['panns_cnn14', 'accuracy_mean'] - p.loc[ms, 'accuracy_mean']):.2f} pp lower** "
        "accuracy — the two frontier points are close in cost and far apart in "
        "accuracy.",
    ]
    for m in MODEL_ORDER:
        lr = 100 * (1 - e32.loc[m, "latency_ms_per_clip_median"]
                    / e1.loc[m, "latency_ms_per_clip_median"])
        er = 100 * (1 - e32.loc[m, "j_per_clip_dynamic"] / e1.loc[m, "j_per_clip_dynamic"])
        L.append(f"- Batching 1 -> 32 changes {DISPLAY[m]} latency by "
                 f"**{-lr:+.1f}%** and dynamic energy by **{-er:+.1f}%** "
                 f"(negative = cheaper at batch 32).")

    L += ["", "## F. FP16 compatibility", ""]
    for m in MODEL_ORDER:
        rows16 = eff[(eff["model"] == m) & (eff["dtype"] == "fp16")]
        if set(rows16["status"]) == {"ok"}:
            r1 = rows16[rows16["batch_size"] == 1].iloc[0]
            dl = 100 * (1 - r1["latency_ms_per_clip_median"]
                        / e1.loc[m, "latency_ms_per_clip_median"])
            de = 100 * (1 - r1["j_per_clip_dynamic"] / e1.loc[m, "j_per_clip_dynamic"])
            L.append(f"- **{DISPLAY[m]}**: fp16 measured. At batch 1, latency "
                     f"{-dl:+.1f}% and dynamic energy {-de:+.1f}% versus fp32.")
        else:
            note = rows16["note"].dropna().iloc[0] if len(rows16["note"].dropna()) else "failed"
            L.append(f"- **{DISPLAY[m]}**: fp16 produced no valid measurement — "
                     f"`{note}` at both batch sizes. No fp16 number is reported for "
                     "this model; the failure is kept explicitly in the CSV.")
    n_ok16 = eff[(eff["dtype"] == "fp16") & (eff["status"] == "ok")]["model"].nunique()
    L += ["", f"fp16 usability is model/implementation dependent: **{n_ok16} of "
          f"{len(MODEL_ORDER)}** models produced finite fp16 embeddings under "
          "`model.half()` with no other change.", ""]

    L += ["## G. Limitations of the current evidence", "",
          "- Single dataset (ESC-50, 2000 clips, 50 classes). UrbanSound8K and TAU "
          "are not yet run, so nothing here generalises across datasets.",
          "- Single GPU type (one Tesla T4 instance). Latency and energy are "
          "hardware-specific and must not be presented as device-independent.",
          "- Energy and latency come from one measurement session per "
          "configuration; the CSV records three latency repeats but a single "
          "energy window per configuration, so no run-to-run energy variance is "
          "available.",
          "- fp16 failures are reported as measured failures of "
          "`model.half()` without further remediation; they are not evidence that "
          "these models cannot be quantised by other means.",
          "- Zero-shot rows are `weak_zero_shot=True`: pretraining-corpus overlap "
          "with ESC-50 sources cannot be excluded.",
          "- No statistical significance testing was performed. Fold dispersion "
          "is reported as a standard deviation over five folds and nothing more; "
          "no claim of significance is supported by these files.",
          "- The probe is deterministic (lbfgs), so a single result exists per "
          "official fold; the dispersion is across folds, not across seeds.", ""]

    L += ["## H. Strongest evidence-backed findings", ""]
    top = probe.iloc[0]
    findings = [
        f"On ESC-50 frozen linear probes, {DISPLAY[top['model']]} leads at "
        f"{pct(top['accuracy_mean'])} +/- {pct(top['accuracy_std'])}% accuracy, "
        f"and the five models span only {spread:.2f} pp "
        f"({pct(worst['accuracy_mean'])}-{pct(best['accuracy_mean'])}%).",
        f"Deployment cost varies far more than accuracy: {DISPLAY[ms]} and "
        f"{DISPLAY[ast_]} differ by only "
        f"{100 * (p.loc[ms, 'accuracy_mean'] - p.loc[ast_, 'accuracy_mean']):.2f} pp "
        f"in accuracy but by "
        f"{e1.loc[ast_, 'latency_ms_per_clip_median'] / e1.loc[ms, 'latency_ms_per_clip_median']:.2f}x "
        f"in latency and "
        f"{e1.loc[ast_, 'j_per_clip_dynamic'] / e1.loc[ms, 'j_per_clip_dynamic']:.2f}x "
        "in dynamic energy per clip at fp32 batch 1 — accuracy-only comparisons "
        "hide this axis entirely.",
        f"The accuracy-energy Pareto frontier at fp32 batch 1 contains exactly "
        f"{len(par[(par['comparison'] == 'B') & par['pareto_optimal']])} of five "
        f"models: "
        f"{', '.join(par[(par['comparison'] == 'B') & par['pareto_optimal']]['model_display'])}; "
        "the other three are strictly dominated.",
        f"Batching from 1 to 32 clips helps unevenly: "
        f"{DISPLAY['panns_cnn14']} gains "
        f"{100 * (1 - e32.loc['panns_cnn14', 'latency_ms_per_clip_median'] / e1.loc['panns_cnn14', 'latency_ms_per_clip_median']):.0f}% "
        f"latency and "
        f"{100 * (1 - e32.loc['panns_cnn14', 'j_per_clip_dynamic'] / e1.loc['panns_cnn14', 'j_per_clip_dynamic']):.0f}% "
        f"dynamic energy per clip, whereas {DISPLAY[ast_]} gets "
        f"{100 * (e32.loc[ast_, 'latency_ms_per_clip_median'] / e1.loc[ast_, 'latency_ms_per_clip_median'] - 1):.0f}% "
        "SLOWER per clip at batch 32.",
        f"fp16 is not uniformly available: {n_ok16} of {len(MODEL_ORDER)} models "
        f"produced finite embeddings under `model.half()`; "
        f"{', '.join(sorted(DISPLAY[m] for m in eff[(eff['dtype'] == 'fp16') & (eff['status'] != 'ok')]['model'].unique()))} "
        "returned NaN/Inf and yield no fp16 measurement.",
        f"Where fp16 does work it is a large win: {DISPLAY[ast_]} at batch 1 drops "
        f"{100 * (1 - float(eff[(eff['model'] == ast_) & (eff['dtype'] == 'fp16') & (eff['batch_size'] == 1)]['latency_ms_per_clip_median'].iloc[0]) / e1.loc[ast_, 'latency_ms_per_clip_median']):.0f}% "
        f"in latency and "
        f"{100 * (1 - float(eff[(eff['model'] == ast_) & (eff['dtype'] == 'fp16') & (eff['batch_size'] == 1)]['j_per_clip_dynamic'].iloc[0]) / e1.loc[ast_, 'j_per_clip_dynamic']):.0f}% "
        "in dynamic energy per clip.",
        "Frozen probes beat zero-shot for both CLAP models on ESC-50: "
        + "; ".join(
            f"{DISPLAY[m]} {pct(p.loc[m, 'accuracy_mean'])}% probe vs "
            f"{pct(zsum[zsum['model'] == m]['accuracy_mean'].max())}% best-template "
            f"zero-shot (+{100 * (p.loc[m, 'accuracy_mean'] - zsum[zsum['model'] == m]['accuracy_mean'].max()):.2f} pp)"
            for m in ZS_MODELS) + ".",
        "Prompt wording matters unequally across CLAP implementations: "
        + "; ".join(
            f"{DISPLAY[m]} shifts "
            f"{100 * (zsum[zsum['model'] == m]['accuracy_mean'].max() - zsum[zsum['model'] == m]['accuracy_mean'].min()):.2f} pp "
            "between the two templates" for m in ZS_MODELS)
        + " (both rows flagged weak zero-shot).",
    ]
    L += [f"{i}. {f}" for i, f in enumerate(findings, 1)]
    L += ["", "---", "", "Provenance: probes/zero-shot at commit "
          f"`{acc['git_commit'].iloc[0]}`, efficiency at commit "
          f"`{eff['git_commit'].iloc[0]}` (the latter differs only by the NVML "
          "sampler thread-lifecycle fix; measurement methodology is identical). "
          f"All rows recorded `git_dirty=False` on {EXPECTED_GPU}.", ""]
    (REPORTS / "M7_ESC50_FINDINGS.md").write_text("\n".join(L) + "\n")
    print("[summarize] wrote reports/M7_ESC50_FINDINGS.md")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    acc = pd.read_csv(RESULTS / "accuracy.csv")
    zs = pd.read_csv(RESULTS / "zeroshot.csv")
    eff = pd.read_csv(RESULTS / "efficiency.csv")

    checks = Checks()
    validate_accuracy(acc, checks)
    validate_zeroshot(zs, checks)
    validate_efficiency(eff, checks)
    write_validation_report(checks, {"accuracy.csv": acc, "zeroshot.csv": zs,
                                     "efficiency.csv": eff})
    if checks.failures:
        for f in checks.failures:
            print(f"VALIDATION FAILURE [{f['group']}] {f['check']}: "
                  f"expected {f['expected']}, observed {f['observed']}")
        raise SystemExit("validation failed — see reports/M7_RESULTS_VALIDATION.md")
    print(f"[summarize] validation PASSED ({len(checks.rows)} checks)")

    probe = probe_summary(acc)
    probe.to_csv(TABLES / "esc50_linear_probe_summary.csv", index=False)
    write_md_table(TABLES / "esc50_linear_probe_summary.md", probe,
                   "ESC-50 frozen linear probe (protocol P2)",
                   "Mean and standard deviation ACROSS THE FIVE OFFICIAL FOLDS "
                   "(fold dispersion, not seed variance). Source: "
                   "`results/accuracy.csv`.")

    zsum = zeroshot_summary(zs)
    zsum.to_csv(TABLES / "esc50_zeroshot_summary.csv", index=False)
    write_md_table(TABLES / "esc50_zeroshot_summary.md", zsum,
                   "ESC-50 zero-shot CLAP (protocol P1)",
                   "Mean and standard deviation across the five official folds. "
                   "All rows are flagged `weak_zero_shot=True`: CLAP pretraining "
                   "corpora may overlap the ESC-50 sources. Source: "
                   "`results/zeroshot.csv`.")

    esum = efficiency_summary(eff)
    esum.to_csv(TABLES / "esc50_efficiency_summary.csv", index=False)
    write_md_table(TABLES / "esc50_efficiency_summary.md", esum,
                   "ESC-50 efficiency on NVIDIA Tesla T4 (M5)",
                   "Measured region: canonical waveform batch -> embedding "
                   "(model-specific feature extraction included). Latency: 20 "
                   "warm-ups, 100 timed iterations, 3 repeats, median and IQR. "
                   "Energy: NVML board power at 10 Hz over a >=30 s window, "
                   "J/clip divided by actual clips processed; `j_per_clip_dynamic` "
                   "subtracts the idle baseline. Failed configurations are kept "
                   "with their error. Source: `results/efficiency.csv`.")

    main_tbl = main_comparison(probe, eff)
    main_tbl.to_csv(TABLES / "esc50_main_comparison.csv", index=False)
    write_md_table(TABLES / "esc50_main_comparison.md", main_tbl,
                   "ESC-50 main comparison (accuracy x deployment efficiency, fp32)",
                   "fp32 is used as the universal fair comparison: all five models "
                   "have valid fp32 measurements. Energy columns are labelled "
                   "explicitly as dynamic (idle-subtracted) or total. Accuracy and "
                   "macro-F1 are frozen-probe means across the five official folds.")

    par = pareto_table(probe, eff)
    par.to_csv(TABLES / "esc50_pareto_frontier.csv", index=False)
    write_md_table(TABLES / "esc50_pareto_frontier.md", par,
                   "ESC-50 Pareto analysis (fp32)",
                   "A configuration is dominated iff another has >= accuracy AND "
                   "<= cost with at least one strict inequality. Computed "
                   "programmatically from the official CSVs.")

    write_findings_report(probe, zsum, eff, main_tbl, par, acc, zs)

    print("[summarize] tables written to results/tables/")
    for key, group in par.groupby("comparison"):
        front = group[group["pareto_optimal"]]["model_display"].tolist()
        print(f"[summarize] Pareto {key} ({group['comparison_label'].iloc[0]}): {front}")


if __name__ == "__main__":
    main()
