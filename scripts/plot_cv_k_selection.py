"""
Standalone plot script for CV-based K selection results.

Two figures:
  1. CV lppd curve — mean CV lppd per node vs K fitted, for K=1 and K=2 ground-truth datasets.
  2. Kernel recovery grid — inferred vs true kernels for each fitted K, best CV fold selected.

Usage:
    uv run python scripts/plot_cv_k_selection.py [--save figures/cv_k_selection]
"""

import argparse
import glob
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import figstyles

figstyles.apply()

RESULTS = Path(__file__).parent.parent / "results"

SWEEPS = {
    "K=1": (
        "cv_model_sel_k1_gt_20260501", "cv_k1_ground_truth", 1,
        [0.05],
    ),
    "K=2": (
        "cv_model_sel_k2_gt_20260501", "cv_k2_ground_truth", 2,
        [0.02, 0.10],
    ),
}

K_FITS  = [1, 2, 3, 4]
NU_MAX  = 15


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_cv_results(sweep_name: str, dataset_name: str) -> pd.DataFrame:
    rows = []
    base = RESULTS / sweep_name / "synthetic" / dataset_name
    for k_dir in sorted(base.glob("K~*"), key=lambda p: int(p.name.split("~")[1])):
        k = int(k_dir.name.split("~")[1])
        fold_means = [
            pd.read_parquet(p)["mean_log_pred"].iloc[0]
            for p in sorted(k_dir.glob("cv_fold~*/cv_held_out_loglik.parquet"))
        ]
        if not fold_means:
            continue
        fold_means = np.array(fold_means)
        n = len(fold_means)
        rows.append({
            "K":                      k,
            "mean_cv_lppd_per_node":  fold_means.mean(),
            "se_cv_lppd_per_node":    fold_means.std(ddof=1) / np.sqrt(n),
            "n_folds":                n,
        })
    return pd.DataFrame(rows).reset_index(drop=True)


def sc_kernel(beta: float, nu_max: int) -> tuple[np.ndarray, np.ndarray]:
    nu = np.arange(1, nu_max + 1)
    return nu, 1 - (1 - beta) ** nu


def select_best_fold(sweep: str, dataset: str, k_fit: int, nu_max: int = NU_MAX):
    """Return kernel_mode_summary from the fold with maximum component separation."""
    pattern = str(
        RESULTS / sweep / "synthetic" / dataset / f"K~{k_fit}" / "cv_fold~*"
        / "kernel_mode_summary.parquet"
    )
    best_path, best_spread = None, -1.0
    for p in sorted(glob.glob(pattern)):
        df = pd.read_parquet(p)
        nu1 = df[df.nu == 1].groupby("component")["median"].mean()
        spread = float(nu1.max() - nu1.min()) if len(nu1) > 1 else 0.0
        if spread > best_spread:
            best_spread, best_path = spread, p
    if best_path is None:
        return None
    df = pd.read_parquet(best_path)
    return df[df.nu <= nu_max].copy()


# ── Figure 1: CV lppd curve ───────────────────────────────────────────────────

def plot_cv_lppd(save_path: Path | None = None) -> None:
    palette = [figstyles.BLUE, figstyles.RED]
    markers = ["o", "s"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, (label, (sweep, dset, true_k, _)), color, marker in zip(
        axes, SWEEPS.items(), palette, markers
    ):
        df = load_cv_results(sweep, dset)
        if df.empty:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
            continue

        mu  = df["mean_cv_lppd_per_node"].values
        se  = df["se_cv_lppd_per_node"].values
        ks  = df["K"].values

        # 1-SE rule: threshold = max lppd - SE at the best K
        best_idx   = int(np.argmax(mu))
        threshold  = mu[best_idx] - se[best_idx]
        # Most parsimonious K above the threshold
        selected_k = int(ks[np.where(mu >= threshold)[0][0]])

        # Clip y-axis: data range + 1.5 SE padding each side
        pad  = 1.5 * se.max()
        y_lo = min(mu) - pad
        y_hi = max(mu) + pad
        ax.set_ylim(y_lo, y_hi)

        # Mean lppd line + error bars
        ax.errorbar(
            ks, mu, yerr=se,
            color=color, marker=marker, ms=8, lw=2.2,
            capsize=4, capthick=1.4, elinewidth=1.2, zorder=3,
        )
        ax.fill_between(ks, mu - se, mu + se, alpha=0.12, color=color)

        # Only draw threshold if it actually excludes at least one K value
        threshold_visible = np.any(mu < threshold)
        if threshold_visible:
            ax.axhline(threshold, color=color, lw=1.2, ls=":", alpha=0.7, zorder=2)

        # Mark selected K — vertical dashed line + label just inside top of plot
        ax.axvline(selected_k, color=color, lw=1.5, ls="--", alpha=0.6, zorder=2)
        ax.annotate(
            f"K={selected_k}",
            xy=(selected_k, y_hi),
            xytext=(selected_k + 0.12, y_hi - (y_hi - y_lo) * 0.04),
            fontsize=8.5, color=color, alpha=0.9, va="top", ha="left",
        )

        # True K — thin black dashed line
        if true_k != selected_k:
            ax.axvline(true_k, color="black", lw=1.2, ls="--", alpha=0.4, zorder=2)

        ax.set_xlabel("K (mixture components)")
        ax.set_ylabel("Mean CV lppd per node")
        ax.set_title(label, pad=10)
        ax.set_xticks(K_FITS)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path.with_suffix(".lppd.pdf"))
        print(f"Saved {save_path.with_suffix('.lppd.pdf')}")
    plt.show()


# ── Figure 2: Kernel recovery grid ───────────────────────────────────────────

def plot_kernel_recovery(save_path: Path | None = None) -> None:
    GT_INFO = {
        "cv_k1_ground_truth": {
            "sweep":      "cv_model_sel_k1_gt_20260501",
            "betas":      [0.05],
            "row_label":  "K=1 GT  (β=0.05)",
            "true_colors": ["black"],
        },
        "cv_k2_ground_truth": {
            "sweep":      "cv_model_sel_k2_gt_20260501",
            "betas":      [0.02, 0.10],
            "row_label":  "K=2 GT  (β=0.02 / 0.10)",
            "true_colors": ["#1B5E20", "#B71C1C"],
        },
    }

    fig, axes = plt.subplots(2, len(K_FITS), figsize=(16, 7), sharey=False)

    for row_idx, (dset, info) in enumerate(GT_INFO.items()):
        for col_idx, k_fit in enumerate(K_FITS):
            ax = axes[row_idx, col_idx]

            # True kernels
            for beta, tc in zip(info["betas"], info["true_colors"]):
                nu, k = sc_kernel(beta, NU_MAX)
                ax.plot(nu, k, color=tc, lw=2, ls="--", zorder=4,
                        label=f"True β={beta}" if col_idx == 0 else None)

            # Inferred components — sorted by nu=1 median for consistent ordering
            df = select_best_fold(info["sweep"], dset, k_fit)
            if df is not None:
                order = (
                    df[df.nu == 1]
                    .sort_values("median")["component"]
                    .tolist()
                )
                for rank, comp_id in enumerate(order):
                    cdf   = df[df.component == comp_id].sort_values("nu")
                    color = figstyles.COMP_COLORS[rank % len(figstyles.COMP_COLORS)]
                    ax.plot(cdf.nu, cdf["median"], color=color, lw=1.8,
                            marker="o", ms=3.5, zorder=3,
                            label=f"Fit comp {rank+1}" if col_idx == 0 else None)
                    ax.fill_between(cdf.nu, cdf.ci_low, cdf.ci_high,
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
        "--save", metavar="PATH",
        help="Base path for saved figures (suffixes .lppd.png / .kernels.png added).",
    )
    args = parser.parse_args()

    save_path = Path(args.save) if args.save else None

    plot_cv_lppd(save_path)
    plot_kernel_recovery(save_path)


if __name__ == "__main__":
    main()
