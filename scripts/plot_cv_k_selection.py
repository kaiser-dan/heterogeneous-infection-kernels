"""
Standalone plot script for CV-based K selection results.

Reads a pre-computed summary parquet (default: results/cv_summary.parquet)
produced by scripts/summarize_cv_results.py. The plot script does no walking
of results/ -- it only reads the summary and applies display config.

Two figures:
  1. CV lppd curve -- mean CV lppd per node vs K fitted, for K=1 and K=2 GT.
  2. Kernel recovery grid -- inferred vs true kernels for each fitted K.

Usage:
    uv run python scripts/plot_cv_k_selection.py [--summary PATH] [--save BASE]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import figstyles

figstyles.apply()

RESULTS = Path(__file__).parent.parent / "results"
DEFAULT_SUMMARY = RESULTS / "cv_summary.parquet"

# Display config: which datasets to plot, their labels, true betas, and colors.
# Data (lppd, kernel modes) all lives in the summary parquet.
PLOT_CONFIG = {
    "cv_k1_ground_truth": {
        "label":       "K=1",
        "row_label":   "K=1 GT  (β=0.05)",
        "betas":       [0.05],
        "true_colors": ["black"],
    },
    "cv_k2_ground_truth": {
        "label":       "K=2",
        "row_label":   "K=2 GT  (β=0.02 / 0.10)",
        "betas":       [0.02, 0.10],
        "true_colors": ["#1B5E20", "#B71C1C"],
    },
}

K_FITS = [1, 2, 3, 4]
NU_MAX = 15


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(
            f"Summary file not found: {path}\n"
            f"Run: uv run python scripts/summarize_cv_results.py"
        )
    return pd.read_parquet(path)


def cv_lppd_for(summary: pd.DataFrame, dataset: str) -> pd.DataFrame:
    df = summary[(summary["kind"] == "cv_lppd") & (summary["dataset"] == dataset)]
    return df.sort_values("k_fit").reset_index(drop=True)


def kernel_recovery_for(summary: pd.DataFrame, dataset: str, k_fit: int) -> pd.DataFrame | None:
    df = summary[
        (summary["kind"] == "kernel_recovery")
        & (summary["dataset"] == dataset)
        & (summary["k_fit"] == k_fit)
    ]
    return df.reset_index(drop=True) if not df.empty else None


def sc_kernel(beta: float, nu_max: int) -> tuple[np.ndarray, np.ndarray]:
    nu = np.arange(1, nu_max + 1)
    return nu, 1 - (1 - beta) ** nu


# ── Figure 1: CV lppd curve ───────────────────────────────────────────────────

def plot_cv_lppd(summary: pd.DataFrame, save_path: Path | None = None) -> None:
    palette = [figstyles.BLUE, figstyles.RED]
    markers = ["o", "s"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, (dataset, info), color, marker in zip(
        axes, PLOT_CONFIG.items(), palette, markers
    ):
        df = cv_lppd_for(summary, dataset)
        if df.empty:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
            continue

        ks = df["k_fit"].values
        mu = df["mean_cv_lppd_per_node"].values
        se = df["se_cv_lppd_per_node"].values
        true_k = int(df["true_k"].iloc[0])

        # 1-SE rule: threshold = max lppd - SE at the best K
        best_idx = int(np.argmax(mu))
        threshold = mu[best_idx] - se[best_idx]
        selected_k = int(ks[np.where(mu >= threshold)[0][0]])

        pad = 1.5 * se.max()
        y_lo, y_hi = mu.min() - pad, mu.max() + pad
        ax.set_ylim(y_lo, y_hi)

        ax.errorbar(
            ks, mu, yerr=se,
            color=color, marker=marker, ms=8, lw=2.2,
            capsize=4, capthick=1.4, elinewidth=1.2, zorder=3,
        )
        ax.fill_between(ks, mu - se, mu + se, alpha=0.12, color=color)

        if np.any(mu < threshold):
            ax.axhline(threshold, color=color, lw=1.2, ls=":", alpha=0.7, zorder=2)

        ax.axvline(selected_k, color=color, lw=1.5, ls="--", alpha=0.6, zorder=2)
        ax.annotate(
            f"K={selected_k}",
            xy=(selected_k, y_hi),
            xytext=(selected_k + 0.12, y_hi - (y_hi - y_lo) * 0.04),
            fontsize=8.5, color=color, alpha=0.9, va="top", ha="left",
        )

        if true_k != selected_k:
            ax.axvline(true_k, color="black", lw=1.2, ls="--", alpha=0.4, zorder=2)

        ax.set_xlabel("K (mixture components)")
        ax.set_ylabel("Mean CV lppd per node")
        ax.set_title(info["label"], pad=10)
        ax.set_xticks(K_FITS)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path.with_suffix(".lppd.pdf"))
        print(f"Saved {save_path.with_suffix('.lppd.pdf')}")
    plt.show()


# ── Figure 2: Kernel recovery grid ───────────────────────────────────────────

def plot_kernel_recovery(summary: pd.DataFrame, save_path: Path | None = None) -> None:
    fig, axes = plt.subplots(2, len(K_FITS), figsize=(16, 7), sharey=False)

    for row_idx, (dataset, info) in enumerate(PLOT_CONFIG.items()):
        for col_idx, k_fit in enumerate(K_FITS):
            ax = axes[row_idx, col_idx]

            for beta, tc in zip(info["betas"], info["true_colors"]):
                nu, k = sc_kernel(beta, NU_MAX)
                ax.plot(nu, k, color=tc, lw=2, ls="--", zorder=4,
                        label=f"True β={beta}" if col_idx == 0 else None)

            df = kernel_recovery_for(summary, dataset, k_fit)
            if df is not None:
                order = (
                    df[df["nu"] == 1]
                    .sort_values("median")["component"]
                    .tolist()
                )
                for rank, comp_id in enumerate(order):
                    cdf = df[df["component"] == comp_id].sort_values("nu")
                    color = figstyles.COMP_COLORS[rank % len(figstyles.COMP_COLORS)]
                    ax.plot(cdf["nu"], cdf["median"], color=color, lw=1.8,
                            marker="o", ms=3.5, zorder=3,
                            label=f"Fit comp {rank+1}" if col_idx == 0 else None)
                    ax.fill_between(cdf["nu"], cdf["ci_low"], cdf["ci_high"],
                                    color=color, alpha=0.12)
            else:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", fontsize=9, color="gray")

            ax.set_title(f"Fit K={k_fit}", fontsize=10)
            ax.set_xticks([1, 5, 10, 15])
            ax.set_xlim(0.5, NU_MAX + 0.5)
            ax.set_ylim(0, None)

            if col_idx == 0:
                ax.set_ylabel(info["row_label"] + "\nP(infect | ν)", fontsize=9)
                ax.legend(fontsize=8, loc="upper left")
            if row_idx == 1:
                ax.set_xlabel("ν (infected neighbours)")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path.with_suffix(".kernels.pdf"))
        print(f"Saved {save_path.with_suffix('.kernels.pdf')}")
    plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary", default=str(DEFAULT_SUMMARY),
        help=f"Path to CV summary parquet (default: {DEFAULT_SUMMARY}).",
    )
    parser.add_argument(
        "--save", metavar="PATH",
        help="Base path for saved figures (suffixes .lppd.pdf / .kernels.pdf added).",
    )
    args = parser.parse_args()

    summary = load_summary(Path(args.summary))
    save_path = Path(args.save) if args.save else None

    plot_cv_lppd(summary, save_path)
    plot_kernel_recovery(summary, save_path)


if __name__ == "__main__":
    main()
