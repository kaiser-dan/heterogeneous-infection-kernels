"""
Model selection plots: synthetic validation + OO + Twitter (partial).

Usage:
    uv run python scripts/plot_model_selection.py

Saves to figures/03-26-26/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

RESULTS = Path("results")
FIGDIR  = Path("figures/03-26-26")
FIGDIR.mkdir(parents=True, exist_ok=True)

MODEL_LABELS = {
    "model_simple_contagion":                  "SC",
    "model_complex_contagion":                 "CC",
    "non_parametric_monotonic_empirical_bayes": "NP",
}
MODEL_ORDER  = ["SC", "CC", "NP"]
MODEL_COLORS = {"SC": "#4C72B0", "CC": "#DD8452", "NP": "#55A868"}
MODEL_MARKERS = {"SC": "o", "CC": "s", "NP": "^"}
K_VALUES = [1, 2, 3]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dir(dataset_name: str) -> pd.DataFrame:
    base = RESULTS / dataset_name
    rows = []
    for mc_path in sorted(base.rglob("model_comparison.parquet")):
        sf_dir  = mc_path.parent
        k_dir   = sf_dir.parent
        sf_name = sf_dir.name.replace("stan_file~", "")
        label   = MODEL_LABELS.get(sf_name, sf_name[:12])
        try:
            k_fit = int(k_dir.name.replace("K~", ""))
        except ValueError:
            continue
        df = pd.read_parquet(mc_path)
        r  = dict(df.iloc[0])
        r["K_fit"] = k_fit
        r["model"] = label
        rows.append(r)
    return pd.DataFrame(rows)


def load_synth(k_true: int) -> pd.DataFrame:
    """Merge v1 (K_fit=1) with v2 (K_fit=2,3) for K_true=2,3."""
    tag = f"k{k_true}"
    if k_true == 1:
        return load_dir(f"synth_model_sel_k1_20260325")
    # v1 has K_fit=1,2,3 but 1ch/500; take only K_fit=1 from v1
    v1 = load_dir(f"synth_model_sel_{tag}_20260325")
    v1 = v1[v1.K_fit == 1].copy()
    # v2 has K_fit=2,3 with 4ch/1000
    v2 = load_dir(f"synth_model_sel_{tag}_v2_20260325")
    v2 = v2[v2.K_fit.isin([2, 3])].copy()
    return pd.concat([v1, v2], ignore_index=True)


def build_waic_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Return model x K_fit pivot of WAIC values."""
    pivot = df.pivot_table(index="model", columns="K_fit", values="waic", aggfunc="first")
    pivot = pivot.reindex(index=MODEL_ORDER, columns=K_VALUES)
    return pivot


def delta_waic(pivot: pd.DataFrame) -> pd.DataFrame:
    """WAIC relative to best entry in each column (K_fit)."""
    return pivot.subtract(pivot.min(axis=0), axis=1)


# ---------------------------------------------------------------------------
# Figure 1: Synthetic WAIC validation (3-panel, one per K_true)
# ---------------------------------------------------------------------------

def fig_synth():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    fig.suptitle(
        "Synthetic model selection validation -- WAIC vs K_fit\n"
        "(4 chains / 1000 warmup / 1000 samples, CC-generated data)",
        fontsize=11,
    )

    for ax, k_true in zip(axes, [1, 2, 3]):
        df    = load_synth(k_true)
        pivot = build_waic_grid(df)

        for m in MODEL_ORDER:
            if m not in pivot.index:
                continue
            vals = pivot.loc[m]
            ks   = [k for k in K_VALUES if not np.isnan(vals[k])]
            ws   = [vals[k] for k in ks]
            ax.plot(
                ks, ws,
                marker=MODEL_MARKERS[m],
                color=MODEL_COLORS[m],
                label=m,
                linewidth=1.8,
                markersize=7,
            )

        ax.axvline(k_true, color="black", linestyle="--", linewidth=1.2, label=f"K_true={k_true}")
        ax.set_title(f"K_true = {k_true}")
        ax.set_xlabel("K_fit (number of components)")
        ax.set_ylabel("WAIC (lower = better)")
        ax.set_xticks(K_VALUES)
        ax.legend(fontsize=8)

    fig.tight_layout()
    path = FIGDIR / "fig_synth_model_selection.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.show()


# ---------------------------------------------------------------------------
# Figure 2: OO empirical model selection
# ---------------------------------------------------------------------------

def fig_oo():
    df    = load_dir("oo_k123_v2_20260325")
    pivot = build_waic_grid(df)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(
        "OO Sim 165 (kmax=30, disparity backbone) -- WAIC model selection\n"
        "SC / CC / NP at K=1,2,3  |  4 chains / 1000 warmup / 1000 samples",
        fontsize=11,
    )

    # Left: raw WAIC line plot
    ax = axes[0]
    for m in MODEL_ORDER:
        if m not in pivot.index:
            continue
        vals = pivot.loc[m]
        ks   = [k for k in K_VALUES if not np.isnan(vals[k])]
        ws   = [vals[k] for k in ks]
        ax.plot(ks, ws, marker=MODEL_MARKERS[m], color=MODEL_COLORS[m],
                label=m, linewidth=2, markersize=8)
    ax.set_title("Raw WAIC")
    ax.set_xlabel("K (components)")
    ax.set_ylabel("WAIC (lower = better)")
    ax.set_xticks(K_VALUES)
    ax.legend()

    # Right: delta-WAIC (relative to K=1 per model, to show direction of improvement)
    ax = axes[1]
    for m in MODEL_ORDER:
        if m not in pivot.index:
            continue
        vals    = pivot.loc[m]
        base    = vals[1]
        ks      = [k for k in K_VALUES if not np.isnan(vals[k])]
        deltas  = [vals[k] - base for k in ks]
        ax.plot(ks, deltas, marker=MODEL_MARKERS[m], color=MODEL_COLORS[m],
                label=m, linewidth=2, markersize=8)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, label="K=1 baseline")
    ax.set_title("Delta-WAIC from K=1 (negative = improvement)")
    ax.set_xlabel("K (components)")
    ax.set_ylabel("WAIC - WAIC(K=1)")
    ax.set_xticks(K_VALUES)
    ax.legend()

    fig.tight_layout()
    path = FIGDIR / "fig_oo_model_selection.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.show()


# ---------------------------------------------------------------------------
# Figure 3: Twitter K=1 partial + OO comparison bar chart
# ---------------------------------------------------------------------------

def fig_twitter_partial():
    oo_df = load_dir("oo_k123_v2_20260325")
    tw_df = load_dir("twitter_k123_v2_20260325")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Twitter (survival, kmax=20, V_max=1000) -- WAIC partial results\n"
        "K=1 complete; K=2/K=3 pending  |  4 chains / 1000 warmup / 1000 samples",
        fontsize=11,
    )

    # Left: Twitter K=1 bar chart -- all 3 models
    ax = axes[0]
    tw1 = tw_df[tw_df.K_fit == 1].set_index("model")
    x   = np.arange(len(MODEL_ORDER))
    bars = ax.bar(
        x,
        [tw1.loc[m, "waic"] if m in tw1.index else np.nan for m in MODEL_ORDER],
        color=[MODEL_COLORS[m] for m in MODEL_ORDER],
        width=0.55,
        zorder=3,
    )
    for bar, m in zip(bars, MODEL_ORDER):
        if m in tw1.index:
            v = tw1.loc[m, "waic"]
            ax.text(bar.get_x() + bar.get_width() / 2, v + 20, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER)
    ax.set_ylabel("WAIC (lower = better)")
    ax.set_title("Twitter -- K=1 only (K=2, K=3 pending)")
    ax.grid(axis="y", alpha=0.4)

    # Right: OO + Twitter side-by-side at K=1, grouped by model
    ax = axes[1]
    oo1  = oo_df[oo_df.K_fit == 1].set_index("model")
    tw1  = tw_df[tw_df.K_fit == 1].set_index("model")
    x    = np.arange(len(MODEL_ORDER))
    w    = 0.35
    oo_vals = [oo1.loc[m, "waic"] if m in oo1.index else np.nan for m in MODEL_ORDER]
    tw_vals = [tw1.loc[m, "waic"] if m in tw1.index else np.nan for m in MODEL_ORDER]

    b1 = ax.bar(x - w/2, oo_vals, width=w, label="OO (kmax=30)",
                color=[MODEL_COLORS[m] for m in MODEL_ORDER], alpha=0.85, zorder=3)
    b2 = ax.bar(x + w/2, tw_vals, width=w, label="Twitter (kmax=20)",
                color=[MODEL_COLORS[m] for m in MODEL_ORDER], alpha=0.45,
                edgecolor=[MODEL_COLORS[m] for m in MODEL_ORDER], linewidth=1.5, zorder=3)

    for bar, v in zip(b1, oo_vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width()/2, v + 20, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=8)
    for bar, v in zip(b2, tw_vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width()/2, v + 20, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER)
    ax.set_ylabel("WAIC (lower = better)")
    ax.set_title("OO vs Twitter -- K=1 WAIC by model")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)

    fig.tight_layout()
    path = FIGDIR / "fig_twitter_partial.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.show()


# ---------------------------------------------------------------------------
# Figure 4: Combined 2x3 summary (synth + OO + Twitter)
# ---------------------------------------------------------------------------

def fig_combined():
    fig = plt.figure(figsize=(16, 9))
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.38)

    fig.suptitle(
        "Heterogeneous infection kernel model selection -- WAIC  |  SC / CC / NP at K=1,2,3\n"
        "Synthetic (CC-generated), OO Sim 165 (kmax=30), Twitter survival (kmax=20, V_max=1000)\n"
        "4 chains / 1000 warmup / 1000 samples",
        fontsize=10.5, y=0.99,
    )

    # --- Row 0: Synthetic K_true=1,2,3 (cols 0-2) + p_waic panel (col 3) ---
    synth_axes = [fig.add_subplot(gs[0, c]) for c in range(3)]

    for ax, k_true in zip(synth_axes, [1, 2, 3]):
        df    = load_synth(k_true)
        pivot = build_waic_grid(df)
        for m in MODEL_ORDER:
            if m not in pivot.index:
                continue
            vals = pivot.loc[m]
            ks   = [k for k in K_VALUES if not np.isnan(vals[k])]
            ws   = [vals[k] for k in ks]
            ax.plot(ks, ws, marker=MODEL_MARKERS[m], color=MODEL_COLORS[m],
                    label=m, linewidth=1.8, markersize=6)
        ax.axvline(k_true, color="black", linestyle="--", linewidth=1.1)
        ax.set_title(f"Synthetic  K_true={k_true}", fontsize=9)
        ax.set_xlabel("K_fit", fontsize=8)
        ax.set_ylabel("WAIC" if k_true == 1 else "", fontsize=8)
        ax.set_xticks(K_VALUES)
        ax.tick_params(labelsize=8)
        if k_true == 1:
            ax.legend(fontsize=7.5)

    # Synthetic p_waic panel (col 3, row 0) -- shows K_true=2 CC overfitting signal
    ax = fig.add_subplot(gs[0, 3])
    for k_true in [1, 2, 3]:
        df  = load_synth(k_true)
        cc1 = df[(df.model == "CC") & (df.K_fit == 1)]
        if len(cc1):
            ax.bar(k_true, cc1.iloc[0]["p_waic"],
                   color=MODEL_COLORS["CC"], alpha=0.7 - 0.15 * (k_true - 1),
                   label=f"K_true={k_true}")
    ax.set_title("CC K_fit=1  p_waic\n(underfitting signal)", fontsize=9)
    ax.set_xlabel("K_true", fontsize=8)
    ax.set_ylabel("p_waic", fontsize=8)
    ax.set_xticks([1, 2, 3])
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7)

    # --- Row 1: OO (cols 0-1) + Twitter partial (cols 2-3) ---
    ax_oo_raw   = fig.add_subplot(gs[1, 0])
    ax_oo_delta = fig.add_subplot(gs[1, 1])
    ax_tw_bar   = fig.add_subplot(gs[1, 2])
    ax_tw_comp  = fig.add_subplot(gs[1, 3])

    # OO raw WAIC
    oo_df  = load_dir("oo_k123_v2_20260325")
    oo_piv = build_waic_grid(oo_df)
    for m in MODEL_ORDER:
        if m not in oo_piv.index:
            continue
        vals = oo_piv.loc[m]
        ks   = [k for k in K_VALUES if not np.isnan(vals[k])]
        ws   = [vals[k] for k in ks]
        ax_oo_raw.plot(ks, ws, marker=MODEL_MARKERS[m], color=MODEL_COLORS[m],
                       label=m, linewidth=1.8, markersize=6)
    ax_oo_raw.set_title("OO  raw WAIC", fontsize=9)
    ax_oo_raw.set_xlabel("K_fit", fontsize=8)
    ax_oo_raw.set_ylabel("WAIC", fontsize=8)
    ax_oo_raw.set_xticks(K_VALUES)
    ax_oo_raw.legend(fontsize=7.5)
    ax_oo_raw.tick_params(labelsize=8)

    # OO delta-WAIC from K=1
    for m in MODEL_ORDER:
        if m not in oo_piv.index:
            continue
        vals   = oo_piv.loc[m]
        base   = vals[1]
        ks     = [k for k in K_VALUES if not np.isnan(vals[k])]
        deltas = [vals[k] - base for k in ks]
        ax_oo_delta.plot(ks, deltas, marker=MODEL_MARKERS[m], color=MODEL_COLORS[m],
                         label=m, linewidth=1.8, markersize=6)
    ax_oo_delta.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax_oo_delta.set_title("OO  delta-WAIC from K=1", fontsize=9)
    ax_oo_delta.set_xlabel("K_fit", fontsize=8)
    ax_oo_delta.set_ylabel("WAIC - WAIC(K=1)", fontsize=8)
    ax_oo_delta.set_xticks(K_VALUES)
    ax_oo_delta.legend(fontsize=7.5)
    ax_oo_delta.tick_params(labelsize=8)

    # Twitter K=1 bar chart
    tw_df = load_dir("twitter_k123_v2_20260325")
    tw1   = tw_df[tw_df.K_fit == 1].set_index("model")
    x     = np.arange(len(MODEL_ORDER))
    bars  = ax_tw_bar.bar(
        x,
        [tw1.loc[m, "waic"] if m in tw1.index else 0 for m in MODEL_ORDER],
        color=[MODEL_COLORS[m] for m in MODEL_ORDER],
        width=0.55, zorder=3,
        alpha=0.85,
    )
    for bar, m in zip(bars, MODEL_ORDER):
        if m in tw1.index:
            v = tw1.loc[m, "waic"]
            ax_tw_bar.text(bar.get_x() + bar.get_width()/2, v + 15, f"{v:.0f}",
                           ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax_tw_bar.set_xticks(x)
    ax_tw_bar.set_xticklabels(MODEL_ORDER, fontsize=8)
    ax_tw_bar.set_ylabel("WAIC", fontsize=8)
    ax_tw_bar.set_title("Twitter  K=1 WAIC\n(K=2, K=3 pending)", fontsize=9)
    ax_tw_bar.grid(axis="y", alpha=0.4)
    ax_tw_bar.tick_params(labelsize=8)

    # Twitter vs OO p_waic comparison at K=1
    oo1   = oo_df[oo_df.K_fit == 1].set_index("model")
    x     = np.arange(len(MODEL_ORDER))
    w     = 0.35
    oo_pw = [oo1.loc[m, "p_waic"] if m in oo1.index else np.nan for m in MODEL_ORDER]
    tw_pw = [tw1.loc[m, "p_waic"] if m in tw1.index else np.nan for m in MODEL_ORDER]
    b1 = ax_tw_comp.bar(x - w/2, oo_pw, width=w, label="OO",
                        color=[MODEL_COLORS[m] for m in MODEL_ORDER], alpha=0.85, zorder=3)
    b2 = ax_tw_comp.bar(x + w/2, tw_pw, width=w, label="Twitter",
                        color=[MODEL_COLORS[m] for m in MODEL_ORDER], alpha=0.4,
                        edgecolor=[MODEL_COLORS[m] for m in MODEL_ORDER], linewidth=1.5, zorder=3)
    ax_tw_comp.set_xticks(x)
    ax_tw_comp.set_xticklabels(MODEL_ORDER, fontsize=8)
    ax_tw_comp.set_ylabel("p_waic (effective params)", fontsize=8)
    ax_tw_comp.set_title("K=1 p_waic: OO vs Twitter\n(high p_waic = underfitting)", fontsize=9)
    ax_tw_comp.legend(fontsize=7.5)
    ax_tw_comp.grid(axis="y", alpha=0.4)
    ax_tw_comp.tick_params(labelsize=8)

    path = FIGDIR / "fig_model_selection_combined.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.show()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    fig_synth()
    fig_oo()
    fig_twitter_partial()
    fig_combined()
