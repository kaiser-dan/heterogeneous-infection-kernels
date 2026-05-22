"""
Model selection analysis for empirical datasets.
Reads model_comparison.parquet from all K/stan_file sweep outputs and builds
WAIC and BIC_median tables.

Usage:
    uv run python scripts/analyze_model_selection.py --dataset oo_k123_v2_20260325
    uv run python scripts/analyze_model_selection.py --dataset twitter_k123_v2_20260325
    uv run python scripts/analyze_model_selection.py --dataset oo_k123_v2_20260325 --dataset twitter_k123_v2_20260325
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_LABELS = {
    "model_simple_contagion":              "SC",
    "model_complex_contagion":             "CC",
    "non_parametric_monotonic_empirical_bayes": "NP",
}

K_VALUES = [1, 2, 3]


def load_dataset_results(results_root: Path, dataset_name: str) -> pd.DataFrame:
    """Load all model_comparison.parquet files for a sweep config by globbing."""
    base = results_root / dataset_name
    if not base.exists():
        print(f"  WARNING: results dir not found: {base}")
        return pd.DataFrame()

    mc_files = sorted(base.rglob("model_comparison.parquet"))
    if not mc_files:
        print(f"  WARNING: no model_comparison.parquet found under {base}")
        return pd.DataFrame()

    rows = []
    for mc_path in mc_files:
        # Path pattern: .../K~<k>/stan_file~<name>/model_comparison.parquet
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
        r["dataset"] = dataset_name
        r["K_fit"]   = k_fit
        r["model"]   = label

        # Pull R-hat max and ESS min from inference_summary if available
        summary_path = sf_dir / "inference_summary.parquet"
        if summary_path.exists():
            try:
                s = pd.read_parquet(summary_path)
                rhat_col = [c for c in s.columns if "r_hat" in c.lower() or "rhat" in c.lower()]
                ess_col  = [c for c in s.columns if "ess_bulk" in c.lower() or c.lower() == "ess"]
                if rhat_col:
                    r["max_rhat"] = float(s[rhat_col[0]].dropna().max())
                if ess_col:
                    r["min_ess"] = float(s[ess_col[0]].dropna().min())
            except Exception:
                pass

        rows.append(r)

    return pd.DataFrame(rows)


def print_waic_table(df: pd.DataFrame, dataset: str):
    print(f"\n{'='*65}")
    print(f"WAIC TABLE  --  {dataset}  (lower = better, * = winner)")
    print(f"{'='*65}")
    print(f"  {'model':>4}  {'K=1':>10}  {'K=2':>10}  {'K=3':>10}  {'winner':>6}  {'margin':>8}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*6}  {'-'*8}")
    for m in ["SC", "CC", "NP"]:
        sub = df[df.model == m].sort_values("K_fit")
        if len(sub) == 0:
            continue
        w = sub["waic"].values
        if len(w) < 3:
            vals = {int(row["K_fit"]): row["waic"] for _, row in sub.iterrows()}
            w = np.array([vals.get(k, np.nan) for k in K_VALUES])
        best = int(np.nanargmin(w))
        sorted_w = sorted(w[~np.isnan(w)])
        margin = sorted_w[1] - sorted_w[0] if len(sorted_w) > 1 else float("nan")
        tagged = [
            ("*" if i == best else " ") + (f"{v:.1f}" if not np.isnan(v) else "  nan")
            for i, v in enumerate(w)
        ]
        print(f"  {m:>4}  {tagged[0]:>10}  {tagged[1]:>10}  {tagged[2]:>10}  {best+1:>6}  {margin:>8.1f}")


def print_bic_median_table(df: pd.DataFrame, dataset: str):
    col = "bic_median" if "bic_median" in df.columns else "bic"
    label = "BIC_MEDIAN" if col == "bic_median" else "BIC (max_ll)"
    print(f"\n{'='*65}")
    print(f"{label} TABLE  --  {dataset}  (lower = better, * = winner)")
    print(f"{'='*65}")
    print(f"  {'model':>4}  {'K=1':>10}  {'K=2':>10}  {'K=3':>10}  {'winner':>6}  {'margin':>8}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*6}  {'-'*8}")
    for m in ["SC", "CC", "NP"]:
        sub = df[df.model == m].sort_values("K_fit")
        if len(sub) == 0:
            continue
        b = sub[col].values
        if len(b) < 3:
            vals = {int(row["K_fit"]): row[col] for _, row in sub.iterrows()}
            b = np.array([vals.get(k, np.nan) for k in K_VALUES])
        best = int(np.nanargmin(b))
        sorted_b = sorted(b[~np.isnan(b)])
        margin = sorted_b[1] - sorted_b[0] if len(sorted_b) > 1 else float("nan")
        tagged = [
            ("*" if i == best else " ") + (f"{v:.1f}" if not np.isnan(v) else "  nan")
            for i, v in enumerate(b)
        ]
        print(f"  {m:>4}  {tagged[0]:>10}  {tagged[1]:>10}  {tagged[2]:>10}  {best+1:>6}  {margin:>8.1f}")


def print_diagnostics(df: pd.DataFrame, dataset: str):
    print(f"\n{'='*65}")
    print(f"DIAGNOSTICS  --  {dataset}")
    print(f"{'='*65}")
    print(f"  {'model':>4}  {'K':>3}  {'lppd':>10}  {'p_waic':>8}  {'med_ll':>10}  {'n_samples':>10}  {'max_rhat':>9}  {'min_ess':>9}")
    for m in ["SC", "CC", "NP"]:
        for k in K_VALUES:
            row = df[(df.model == m) & (df.K_fit == k)]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            lppd    = f"{r['lppd']:.1f}"     if "lppd"            in r and not np.isnan(r["lppd"])            else "  nan"
            pwaic   = f"{r['p_waic']:.1f}"   if "p_waic"          in r and not np.isnan(r["p_waic"])          else "  nan"
            medll   = f"{r['median_log_lik']:.1f}" if "median_log_lik" in r and not np.isnan(r["median_log_lik"]) else "  nan"
            nsamp   = f"{int(r['n_samples'])}" if "n_samples"      in r                                        else "  nan"
            rhat    = f"{r['max_rhat']:.3f}"  if "max_rhat"        in r and not np.isnan(r.get("max_rhat", float('nan'))) else "    --"
            ess     = f"{r['min_ess']:.0f}"   if "min_ess"         in r and not np.isnan(r.get("min_ess",  float('nan'))) else "    --"
            print(f"  {m:>4}  {k:>3}  {lppd:>10}  {pwaic:>8}  {medll:>10}  {nsamp:>10}  {rhat:>9}  {ess:>9}")


def summarize_selection(df: pd.DataFrame, dataset: str):
    print(f"\n--- SELECTION SUMMARY: {dataset} ---")
    waic_col = "waic"
    bic_col  = "bic_median" if "bic_median" in df.columns else "bic"
    for m in ["SC", "CC", "NP"]:
        sub = df[df.model == m].sort_values("K_fit")
        if len(sub) < 3:
            print(f"  {m}: incomplete ({len(sub)}/3 K values)")
            continue
        waic_winner = int(sub.iloc[sub[waic_col].values.argmin()]["K_fit"])
        bic_winner  = int(sub.iloc[sub[bic_col].values.argmin()]["K_fit"])
        w = sub[waic_col].values
        waic_margin = sorted(w)[1] - sorted(w)[0]
        b = sub[bic_col].values
        bic_margin  = sorted(b)[1] - sorted(b)[0]
        print(f"  {m}:  WAIC -> K={waic_winner} (margin {waic_margin:.1f})   {bic_col} -> K={bic_winner} (margin {bic_margin:.1f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", required=True,
                        help="Config name (e.g. oo_k123_v2_20260325). Repeat for multiple.")
    parser.add_argument("--results-dir", default="results",
                        help="Root results directory (default: results/)")
    args = parser.parse_args()

    results_root = Path(args.results_dir)

    for dataset in args.dataset:
        print(f"\n{'#'*65}")
        print(f"# Dataset: {dataset}")
        print(f"{'#'*65}")

        df = load_dataset_results(results_root, dataset)
        if df.empty:
            print("  No data found.")
            continue

        print(f"  Loaded {len(df)} model_comparison entries")

        print_waic_table(df, dataset)
        print_bic_median_table(df, dataset)
        print_diagnostics(df, dataset)
        summarize_selection(df, dataset)


if __name__ == "__main__":
    main()
