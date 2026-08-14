#!/usr/bin/env python3
"""M7 — publication figures, generated deterministically from the official CSVs.

Inputs:  results/accuracy.csv, results/zeroshot.csv, results/efficiency.csv
Outputs: figures/*.pdf (vector, for the paper) and figures/*.png (300 dpi)

Styling is centralised in scripts/fig_style.py (Okabe-Ito palette, fixed
model->color mapping, Type-42 fonts, IEEE column width). No seaborn, no random
state, no fabricated values: failed fp16 configurations are annotated, never
plotted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from fig_style import (  # noqa: E402
    BAR_EDGE, BAR_H, COL_W, INK, MARKERS, MODEL_COLORS, NOTE_GRAY, PAIR_COLORS,
    SCATTER_H, TEXT_W, apply_rcparams, save, style_bar_axes, style_scatter_axes,
)
from summarize_results import (  # noqa: E402
    DISPLAY, MODEL_ORDER, pareto_front, probe_summary,
)

RESULTS = ROOT / "results"

PARETO_MS = 8.0      # Pareto-optimal markers
DOMINATED_MS = 7.0

apply_rcparams()


def load():
    acc = pd.read_csv(RESULTS / "accuracy.csv")
    zs = pd.read_csv(RESULTS / "zeroshot.csv")
    eff = pd.read_csv(RESULTS / "efficiency.csv")
    probe = probe_summary(acc).set_index("model")
    return probe, zs, eff


def fp32(eff: pd.DataFrame, bs: int) -> pd.DataFrame:
    sel = eff[(eff["dtype"] == "fp32") & (eff["batch_size"] == bs)
              & (eff["status"] == "ok")].copy()
    sel["model"] = pd.Categorical(sel["model"], MODEL_ORDER, ordered=True)
    return sel.sort_values("model").reset_index(drop=True)


def scatter_pareto(ax, data: pd.DataFrame, cost_col: str, probe: pd.DataFrame,
                   xlabel: str, label_offsets: dict[str, tuple[float, float, str]],
                   xticks: list[float]):
    """Accuracy (%) vs a cost metric, Pareto-optimal points marked.

    One Okabe-Ito color per model; filled marker = Pareto-optimal,
    hollow (white face, colored edge) = dominated.

    label_offsets: model -> (x multiplier, y offset in accuracy points,
    horizontal alignment) so every label sits beside its own marker.
    """
    data = data.copy()
    data["accuracy_pct"] = 100 * data["model"].map(probe["accuracy_mean"]).astype(float)
    data["optimal"] = pareto_front(data, "accuracy_pct", cost_col)

    for row in data.itertuples():
        opt = row.optimal
        colour = MODEL_COLORS[row.model]
        ax.plot(getattr(row, cost_col), row.accuracy_pct,
                marker=MARKERS[row.model],
                markersize=PARETO_MS if opt else DOMINATED_MS,
                markerfacecolor=colour if opt else "white",
                markeredgecolor=colour,
                markeredgewidth=1.2, linestyle="none",
                zorder=3 if opt else 2)
        dx, dy, ha = label_offsets.get(row.model, (1.06, 0.0, "left"))
        ax.annotate(DISPLAY[row.model],
                    (getattr(row, cost_col), row.accuracy_pct),
                    xytext=(getattr(row, cost_col) * dx, row.accuracy_pct + dy),
                    fontsize=7, va="center", ha=ha, color=INK, zorder=4)

    ax.annotate("filled marker = Pareto-optimal", xy=(0.97, 0.04),
                xycoords="axes fraction", fontsize=6.5, color=NOTE_GRAY,
                ha="right")
    style_scatter_axes(ax)
    ax.set_xscale("log")
    # explicit ticks: the default log minor labels collide at this width
    ax.set_xticks(xticks)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(xlabel)
    ax.set_ylabel("ESC-50 accuracy (%)")
    return data


def figure1(probe, eff):
    """Accuracy vs dynamic (idle-subtracted) energy, fp32, batch 1."""
    fig, ax = plt.subplots(figsize=(COL_W, SCATTER_H))
    data = scatter_pareto(
        ax, fp32(eff, 1), "j_per_clip_dynamic", probe,
        "Dynamic energy per clip (J, log scale)\nNVML board power, idle baseline subtracted",
        {"panns_cnn14": (1.08, -0.30, "left"), "ms_clap": (1.08, 0.30, "left"),
         "beats": (1.08, 0.0, "left"), "laion_clap": (1.08, 0.0, "left"),
         "ast": (0.93, 0.0, "right")},
        xticks=[0.6, 0.8, 1.0, 2.0, 3.0, 5.0, 7.0])
    ax.set_ylim(91.0, 99.3)
    ax.set_xlim(0.55, 8.0)
    save(fig, "fig1_accuracy_vs_energy_fp32_bs1")
    return data


def figure2(probe, eff):
    """Accuracy vs median latency, fp32, batch 1."""
    fig, ax = plt.subplots(figsize=(COL_W, SCATTER_H))
    data = scatter_pareto(
        ax, fp32(eff, 1), "latency_ms_per_clip_median", probe,
        "Median latency per clip (ms, log scale)\nwaveform to embedding, batch size 1",
        {"panns_cnn14": (1.10, -0.30, "left"), "ms_clap": (1.10, 0.30, "left"),
         "beats": (1.10, 0.0, "left"), "laion_clap": (1.10, 0.0, "left"),
         "ast": (0.90, 0.0, "right")},
        xticks=[10, 15, 20, 30, 50, 80, 120])
    ax.set_ylim(91.0, 99.3)
    ax.set_xlim(9.5, 130.0)
    save(fig, "fig2_accuracy_vs_latency_fp32_bs1")
    return data


BATCH_NOTE = "annotation = batch-1 / batch-32 ratio (>1 favours batching)"


def _batch_panel(ax, merged: pd.DataFrame, metric: str, ylabel: str) -> None:
    """One grouped-bar panel of the batch-size comparison (batch 1 vs 32)."""
    x = range(len(merged))
    width = 0.38
    v1 = merged[f"{metric}_b1"].to_numpy()
    v32 = merged[f"{metric}_b32"].to_numpy()
    ax.bar([i - width / 2 for i in x], v1, width, label="batch 1",
           color=PAIR_COLORS[0], edgecolor=BAR_EDGE, linewidth=0.5)
    ax.bar([i + width / 2 for i in x], v32, width, label="batch 32",
           color=PAIR_COLORS[1], edgecolor=BAR_EDGE, linewidth=0.5)
    for i, (a, b) in enumerate(zip(v1, v32)):
        ax.annotate(f"{a / b:.2f}x", (i, max(a, b)), textcoords="offset points",
                    xytext=(0, 2.5), ha="center", fontsize=6.5, color=INK)
    ax.set_xticks(list(x))
    ax.set_xticklabels([DISPLAY[m] for m in merged["model"]], rotation=20,
                       ha="right")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(v1.max(), v32.max()) * 1.18)
    style_bar_axes(ax)


def figure3(eff):
    """Batch-size effect on per-clip latency and dynamic energy, fp32.

    Two-panel version kept for the repo; the paper uses the native
    single-panel figure produced by figure3_single().
    """
    b1, b32 = fp32(eff, 1), fp32(eff, 32)
    merged = b1.merge(b32, on="model", suffixes=("_b1", "_b32"))

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.6))
    _batch_panel(axes[0], merged, "latency_ms_per_clip_median",
                 "Median latency per clip (ms)")
    _batch_panel(axes[1], merged, "j_per_clip_dynamic",
                 "Dynamic energy per clip (J)")
    axes[0].legend(loc="upper right", frameon=False)
    axes[0].annotate(BATCH_NOTE, xy=(0.0, 1.02), xycoords="axes fraction",
                     fontsize=6.5, color=NOTE_GRAY)
    fig.tight_layout()
    save(fig, "fig3_batch_effect_fp32")
    return merged


def figure3_single(merged: pd.DataFrame):
    """Native single-panel latency version of fig3, used in the paper
    (replaces the temporary crop of the two-panel figure)."""
    fig, ax = plt.subplots(figsize=(COL_W, BAR_H))
    _batch_panel(ax, merged, "latency_ms_per_clip_median",
                 "Median latency per clip (ms)")
    ax.legend(loc="upper right", frameon=False)
    ax.annotate(BATCH_NOTE, xy=(0.0, 1.02), xycoords="axes fraction",
                fontsize=6.5, color=NOTE_GRAY)
    save(fig, "fig3_batch_latency")


def figure4(zs):
    """Zero-shot prompt sensitivity, mean +/- std across official folds."""
    g = (zs.groupby(["model", "template_id"])
         .agg(mean=("accuracy", "mean"), std=("accuracy", "std"))
         .reset_index())
    models = ["laion_clap", "ms_clap"]
    templates = ["primary", "alternative"]
    template_text = {t: zs[zs["template_id"] == t]["prompt_template"].iloc[0]
                     for t in templates}
    width = 0.35

    fig, ax = plt.subplots(figsize=(COL_W, BAR_H))
    for k, t in enumerate(templates):
        vals, errs = [], []
        for m in models:
            r = g[(g["model"] == m) & (g["template_id"] == t)].iloc[0]
            vals.append(100 * r["mean"])
            errs.append(100 * r["std"])
        pos = [i + (k - 0.5) * width for i in range(len(models))]
        ax.bar(pos, vals, width, yerr=errs, capsize=2, edgecolor=BAR_EDGE,
               linewidth=0.5, color=PAIR_COLORS[k],
               label=f'{t}: "{template_text[t]}"',
               error_kw=dict(elinewidth=0.8, capthick=0.8, ecolor=INK))
        for p, v, e in zip(pos, vals, errs):
            ax.annotate(f"{v:.2f}", (p, v + e), textcoords="offset points",
                        xytext=(0, 2.5), ha="center", fontsize=6.5, color=INK)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([DISPLAY[m] for m in models])
    ax.set_ylabel("Zero-shot accuracy (%)")
    ax.set_ylim(70, 100)
    style_bar_axes(ax)
    # legend outside the plotting area: at this width it would otherwise sit
    # on top of the LAION-CLAP bars
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False,
              fontsize=6.3, ncol=1, handlelength=1.6, borderpad=0.2,
              labelspacing=0.3)
    ax.set_xlabel("error bars: std across the 5 official folds; weak zero-shot "
                  "(possible pretraining overlap)", fontsize=6, color=NOTE_GRAY,
                  labelpad=6)
    save(fig, "fig4_zeroshot_prompt_sensitivity")
    return g


def figure5(eff):
    """fp32 vs fp16 where fp16 actually produced valid measurements.

    NOTE: this figure is NOT included in the SIDe'26 submission — Table III
    carries the precision-ablation data. Kept restyled for repo consistency.
    """
    ok16 = eff[(eff["dtype"] == "fp16") & (eff["status"] == "ok")]["model"].unique()
    failed = sorted({DISPLAY[m] for m in
                     eff[(eff["dtype"] == "fp16") & (eff["status"] != "ok")]["model"]})
    valid = [m for m in MODEL_ORDER if m in set(ok16)]

    labels, lat32, lat16, e32, e16 = [], [], [], [], []
    for m in valid:
        for bs in (1, 32):
            labels.append(f"{DISPLAY[m]}\nbatch {bs}")
            for dt, lat, ener in (("fp32", lat32, e32), ("fp16", lat16, e16)):
                r = eff[(eff["model"] == m) & (eff["dtype"] == dt)
                        & (eff["batch_size"] == bs) & (eff["status"] == "ok")].iloc[0]
                lat.append(float(r["latency_ms_per_clip_median"]))
                ener.append(float(r["j_per_clip_dynamic"]))

    x = range(len(labels))
    width = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.7))
    for ax, v32, v16, ylabel in (
        (axes[0], lat32, lat16, "Median latency per clip (ms)"),
        (axes[1], e32, e16, "Dynamic energy per clip (J)"),
    ):
        ax.bar([i - width / 2 for i in x], v32, width, label="fp32",
               color=PAIR_COLORS[0], edgecolor=BAR_EDGE, linewidth=0.5)
        ax.bar([i + width / 2 for i in x], v16, width, label="fp16",
               color=PAIR_COLORS[1], edgecolor=BAR_EDGE, linewidth=0.5)
        for i, (a, b) in enumerate(zip(v32, v16)):
            # signed change of fp16 relative to fp32: negative = fp16 cheaper
            ax.annotate(f"{100 * (b - a) / a:+.0f}%", (i, max(a, b)),
                        textcoords="offset points", xytext=(0, 2.5), ha="center",
                        fontsize=6.5, color=INK)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=6.5)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(max(v32), max(v16)) * 1.20)
        style_bar_axes(ax)
    axes[0].legend(loc="upper right", frameon=False)
    axes[0].annotate("annotation = fp16 change vs fp32 (negative = cheaper)",
                     xy=(0.0, 1.03), xycoords="axes fraction", fontsize=6.3,
                     color=NOTE_GRAY)
    axes[1].annotate(
        "fp16 unmeasurable (NaN/Inf in embeddings): " + ", ".join(failed),
        xy=(0.0, 1.03), xycoords="axes fraction", fontsize=6.3, color=NOTE_GRAY)
    fig.tight_layout()
    save(fig, "fig5_precision_effect")
    return valid, failed


def main() -> None:
    probe, zs, eff = load()
    figure1(probe, eff)
    figure2(probe, eff)
    merged = figure3(eff)
    figure3_single(merged)
    figure4(zs)
    figure5(eff)
    print("[figures] done")


if __name__ == "__main__":
    main()
