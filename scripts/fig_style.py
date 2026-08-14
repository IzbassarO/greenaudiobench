"""Shared figure style for GreenAudioBench (IEEE two-column, SIDe'26).

Single source of truth for palette, marker shapes, rcParams, axis styling and
physical figure sizes. Figures are inserted at \\columnwidth = 3.5 in, so they
are rendered at final physical size — fonts are never scaled down in LaTeX.

Palette: Okabe-Ito (colorblind-safe). The model->color mapping is fixed and
identical in every figure. Yellow (#F0E442) is excluded: it is illegible on a
white surface at these mark sizes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

FIGURES = Path(__file__).resolve().parents[1] / "figures"

# ---------------------------------------------------------------------------
# Okabe-Ito palette and fixed model mapping (order: MS-CLAP, LAION-CLAP, AST,
# BEATs, PANNs CNN14 — keep this assignment consistent everywhere).
# ---------------------------------------------------------------------------
OKABE_ITO = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}

MODEL_COLORS = {
    "ms_clap": OKABE_ITO["orange"],
    "laion_clap": OKABE_ITO["sky_blue"],
    "ast": OKABE_ITO["bluish_green"],
    "beats": OKABE_ITO["blue"],
    "panns_cnn14": OKABE_ITO["vermillion"],
}

#: Two-color pair for grouped-bar condition contrast (batch 1 vs 32,
#: primary vs alternative prompt, fp32 vs fp16). Not a model identity.
PAIR_COLORS = (OKABE_ITO["blue"], OKABE_ITO["orange"])

#: Distinct marker shapes: shape carries identity alongside color, so the
#: figures survive greyscale printing and color-vision deficiency.
MARKERS = {"ast": "o", "laion_clap": "s", "panns_cnn14": "^",
           "beats": "D", "ms_clap": "v"}

#: Text never wears a series color: dark ink for labels, gray for footnotes.
INK = "#333333"
NOTE_GRAY = "0.35"
BAR_EDGE = "#333333"

# ---------------------------------------------------------------------------
# Physical sizes (inches)
# ---------------------------------------------------------------------------
COL_W = 3.5        # \columnwidth
TEXT_W = 7.16      # \textwidth (two-panel repo-only figures)
SCATTER_H = 2.8
BAR_H = 2.5


def apply_rcparams() -> None:
    """IEEE-safe defaults: TrueType (Type-42) fonts for PDF eXpress,
    sans-serif, base size 8, clean open axes."""
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.axisbelow": True, "axes.grid": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "lines.linewidth": 1.0,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def style_bar_axes(ax) -> None:
    """Subtle horizontal guide lines only; no vertical grid on bar charts."""
    ax.grid(axis="y", linewidth=0.4, alpha=0.4)
    ax.grid(axis="x", visible=False)


def style_scatter_axes(ax) -> None:
    """Light grid on both axes for scatter plots."""
    ax.grid(True, linewidth=0.4, alpha=0.25)


def save(fig, stem: str) -> None:
    """Write both vector (paper) and raster (preview) outputs."""
    FIGURES.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{stem}.{ext}")
    plt.close(fig)
    print(f"[figures] wrote figures/{stem}.pdf and .png")
