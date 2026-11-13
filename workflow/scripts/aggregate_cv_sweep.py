"""Aggregate CV-sweep fold files (one row per held-out node) into:

  _cv_curve.parquet      one row per (cell, dgp, rep, K): node-pooled mean/SE LPPD, divergences
  _cv_selection.parquet  one row per (cell, dgp, rep): K selected by the 1-SE rule, whether it is
                         the true K, the paired K=2 vs K=1 LPPD difference, exposure-event totals
  _cv_by_cell.parquet    one row per (cell, dgp): recovery rate, how often each K is selected,
                         mean continuous signal, exposure and divergence summaries

Continuous signal is: z_21 = delta_lppd_21 / se_delta_21, with
(1) the paired difference in per-node held-out LPPD between K=2 and K=1,
delta_lppd_21 = mean_i [lppd_i(K=2) - lppd_i(K=1)]
(2) SE sd_i(...) / sqrt(V)
Positive values of z_21 favor two components. Per cell: mean delta_lppd_21 and median z_21.

Run standalone (ad-hoc reruns outside Snakemake):
    uv run python workflow/scripts/aggregate_cv_sweep.py results cv_epsilon_omega_sweep
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_cv import select_k_1se

N_NONAXIS_WILDCARDS = 4  # dgp, K, rep, fold
SE_FLOOR = 1e-6          # paired-difference SE below this -> z undefined (NaN)


def cv_curve(nodes: pd.DataFrame, rep_keys: list[str]) -> pd.DataFrame:
    curve = (
        nodes.groupby(rep_keys + ["K"], dropna=False)
             .agg(mean_lppd=("log_pred", "mean"), # mean lppd
                  sd_lppd=("log_pred", "std"),    # sd lppd
                  n_nodes=("log_pred", "size"))   # num nodes
             .reset_index()
    )
    curve["se_lppd"] = curve["sd_lppd"] / np.sqrt(curve["n_nodes"])

    # total divergences across K's 5 fits
    div = (
        nodes.drop_duplicates(rep_keys + ["K", "fold_idx"])
             .groupby(rep_keys + ["K"], dropna=False)["n_divergent"].sum()
             .rename("n_divergent")
             .reset_index()
    )
    return curve.merge(div, on=rep_keys + ["K"], how="left")

# computes continuous K=2 vs. K=1 signal
def paired_delta(nodes: pd.DataFrame, rep_keys: list[str], k_hi: int, k_lo: int) -> pd.DataFrame:
    # one row per (cell, dgp, rep, node), one col per K
    wide = nodes.pivot_table(index=rep_keys + ["node_idx"], columns="K", values="log_pred")
    if k_hi not in wide.columns or k_lo not in wide.columns:
        return pd.DataFrame(columns=rep_keys)
    d = (wide[k_hi] - wide[k_lo]).rename("d").reset_index() # paired difference: d_i = lppd_i(K=2) - lppd_i(K=1)
    out = (
        d.groupby(rep_keys, dropna=False)["d"]
         .agg(delta="mean", sd="std", n="size")
         .reset_index()
    )
    tag = f"{k_hi}{k_lo}"
    out[f"delta_lppd_{tag}"] = out["delta"]                # delta_lppd_21 = mean(d)
    out[f"se_delta_{tag}"] = out["sd"] / np.sqrt(out["n"]) # se_delta_21 = sd(d) / sqrt(n)

    # undefined when the two fits score every node (near-)identically, e.g., an extra component
    # that collapsed to negligible weight (a vanishing SE would otherwise blow z up...)
    se = out[f"se_delta_{tag}"]
    out[f"z_{tag}"] = (out[f"delta_lppd_{tag}"] / se).where(se > SE_FLOOR)

    return out.drop(columns=["delta", "sd", "n"])

# one row per (cell, dgp, rep)
def selections(nodes: pd.DataFrame, curve: pd.DataFrame, rep_keys: list[str]) -> pd.DataFrame:
    # select K
    rows = []
    for key, g in curve.groupby(rep_keys, dropna=False):
        rows.append({**dict(zip(rep_keys, key)), "selected_K": select_k_1se(g)})
    sel = pd.DataFrame(rows)

    # score
    true_k = nodes.drop_duplicates(rep_keys)[rep_keys + ["true_K"]]
    sel = sel.merge(true_k, on=rep_keys, how="left")
    sel["correct"] = sel["selected_K"] == sel["true_K"]

    # exposure totals: every node is held out exactly once per (rep, K), so use one K
    k0 = nodes["K"].min()
    ev = (
        nodes[nodes["K"] == k0]
             .groupby(rep_keys, dropna=False)
             .agg(total_M=("node_M", "sum"), total_N=("node_N", "sum"))
             .reset_index()
    )
    ev["total_events"] = ev["total_M"] + ev["total_N"] # num. timesteps at which some node susceptible with >= 1 infected nbr
    sel = sel.merge(ev, on=rep_keys, how="left")

    div = curve.groupby(rep_keys, dropna=False)["n_divergent"].sum().rename("n_divergent").reset_index()
    sel = sel.merge(div, on=rep_keys, how="left")

    return sel.merge(paired_delta(nodes, rep_keys, 2, 1), on=rep_keys, how="left")

# one row per (cell, dgp)
def by_cell(sel: pd.DataFrame, axes: list[str], k_values) -> pd.DataFrame:
    cell_keys = axes + ["dgp"]
    out = (
        sel.groupby(cell_keys, dropna=False)
           .agg(true_K=("true_K", "first"),
                n_reps=("correct", "size"),
                recovery_rate=("correct", "mean"),
                mean_delta_lppd_21=("delta_lppd_21", "mean"),
                median_z_21=("z_21", "median"),
                mean_total_events=("total_events", "mean"),
                min_total_events=("total_events", "min"),
                mean_total_M=("total_M", "mean"),
                n_divergent=("n_divergent", "sum"))
           .reset_index()
    )
    for k in k_values:
        frac = (
            sel.assign(_hit=sel["selected_K"] == k)
               .groupby(cell_keys, dropna=False)["_hit"].mean()
               .rename(f"frac_selected_K{k}")
               .reset_index()
        )
        out = out.merge(frac, on=cell_keys, how="left")
    return out

# mainly for stand-alone mode
def check_complete(nodes: pd.DataFrame, rep_keys: list[str]) -> None:
    """Every node must appear exactly once per (cell, dgp, rep, K)."""
    dup = nodes.duplicated(rep_keys + ["K", "node_idx"])
    if dup.any():
        raise ValueError(f"{int(dup.sum())} duplicated node records")
    counts = nodes.groupby(rep_keys + ["K"], dropna=False)["node_idx"].agg(["size", "max"])
    bad = counts[counts["size"] != counts["max"] + 1]
    if len(bad):
        raise ValueError(f"incomplete folds for {len(bad)} (cell, dgp, rep, K) groups:\n{bad.head()}")

# driver
def run(files, out_paths, axes: list[str]) -> None:
    files = list(files)
    if not files:
        raise FileNotFoundError("no fold files given")
    nodes = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    rep_keys = axes + ["dgp", "rep"]
    check_complete(nodes, rep_keys)

    curve = cv_curve(nodes, rep_keys)
    sel = selections(nodes, curve, rep_keys)
    cells = by_cell(sel, axes, sorted(curve["K"].unique()))

    curve_path, sel_path, cell_path = (Path(p) for p in out_paths)
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    curve.to_parquet(curve_path, index=False)
    sel.to_parquet(sel_path, index=False)
    cells.to_parquet(cell_path, index=False)

    print(f"pooled {len(nodes)} node records into {len(curve)} (cell, dgp, rep, K) points")
    print(cells.to_string(index=False))


def main():
    results_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    experiment = sys.argv[2] if len(sys.argv) > 2 else "cv_epsilon_omega_sweep"
    out_dir = results_root / "cv" / experiment
    files = sorted((out_dir / "folds").glob("*.parquet"))
    root = Path(__file__).resolve().parent.parent.parent
    wc = json.loads((root / "config" / "params" / f"{experiment}.wildcard_columns.json").read_text())
    outs = [out_dir / n for n in ("_cv_curve.parquet", "_cv_selection.parquet", "_cv_by_cell.parquet")]
    run(files, outs, wc[:-N_NONAXIS_WILDCARDS])


try:
    snakemake
except NameError:
    if __name__ == "__main__":
        main()
else:
    run(snakemake.input, snakemake.output, list(snakemake.params.axes))
