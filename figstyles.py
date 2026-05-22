"""Figure styles for the heterogeneous-infection-kernels project."""

import matplotlib as mpl
import matplotlib.pyplot as plt

# ── Palette ───────────────────────────────────────────────────────────────────
BLUE   = "#2196F3"
RED    = "#E91E63"
ORANGE = "#FF9800"
GREEN  = "#4CAF50"
PURPLE = "#9C27B0"
TEAL   = "#009688"

COLOR_CYCLE = [BLUE, RED, ORANGE, GREEN, PURPLE, TEAL]

# Component colours used in mixture-model kernel plots
COMP_COLORS = COLOR_CYCLE


# ── rcParams ──────────────────────────────────────────────────────────────────
def apply():
    """Apply project-wide matplotlib style. Call once at the top of a script."""
    mpl.rcParams.update({
        "font.family":       "sans-serif",
        "font.size":         11,
        "axes.titlesize":    12,
        "axes.labelsize":    11,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "legend.framealpha": 0.8,

        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    0.8,
        "axes.prop_cycle":   mpl.cycler(color=COLOR_CYCLE),

        "axes.grid":         True,
        "grid.color":        "#e0e0e0",
        "grid.linewidth":    0.5,
        "grid.linestyle":    "--",

        "lines.linewidth":   1.8,
        "lines.markersize":  5,

        "figure.dpi":        120,
        "figure.facecolor":  "white",
        "savefig.dpi":       200,
        "savefig.bbox":      "tight",
        "savefig.facecolor": "white",
    })
