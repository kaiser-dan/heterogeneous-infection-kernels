"""
Aggregate held-out log-likelihoods across CV folds into a single summary row.
Called by workflow/rules/cv.smk rule run_cv_aggregate.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def aggregate_cv_folds(fold_dirs: list[str], output_path: str, group_id: str = "") -> pd.DataFrame:
    """Read cv_held_out_loglik.parquet from each fold directory and summarize.

    Computes mean and standard error of per-node lppd across folds. 

    Parameters
    ----------
    fold_dirs : list of directories, one per CV fold, each containing
                cv_held_out_loglik.parquet written by run_stan_inference.
    output_path : destination for the aggregated cv_summary.parquet.
    group_id : the encoded group_instance_id (e.g. 'K~2__stan_file~mixture_simple_contagion'),
               carried through to the output for joining with other tables.

    Returns
    -------
    Single-row DataFrame written to output_path and returned.
    """
    # Collect one record per fold, warn and skip missing files
    records = []
    for d in fold_dirs:
        p = Path(d) / "cv_held_out_loglik.parquet"
        if not p.exists():
            print(f"Warning: {p} not found — skipping this fold.")
            continue
        records.append(pd.read_parquet(p))

    if not records:
        raise ValueError(f"No cv_held_out_loglik.parquet files found in {fold_dirs}")

    df = pd.concat(records, ignore_index=True)

    # Sanity check: all folds should agree on the total number of folds
    if df["cv_folds"].nunique() != 1:
        raise ValueError(
            f"Inconsistent cv_folds values across fold dirs: {df['cv_folds'].unique().tolist()}"
        )

    expected_cv_folds = int(df["cv_folds"].iloc[0])
    total_lppd        = float(df["cv_score"].sum())
    total_nodes       = int(df["n_held_out"].sum())

    # Per-fold per-node lppd: divide each fold's total cv_score by its node count
    fold_means = df["cv_score"] / df["n_held_out"]
    mean_lppd  = float(fold_means.mean())
    se_lppd    = float(fold_means.std(ddof=1) / np.sqrt(len(records)))

    summary = pd.DataFrame([{
        "cv_folds":              expected_cv_folds,
        "total_cv_lppd":         total_lppd,
        "mean_cv_lppd_per_node": mean_lppd,
        "se_cv_lppd_per_node":   se_lppd,
        "n_total_nodes":         total_nodes,
        "n_folds_found":         len(records),
        "group_instance_id":     group_id,
    }])

    summary.to_parquet(output_path, index=False)
    print(
        f"CV aggregate: total_cv_lppd={total_lppd:.4f}  "
        f"mean_per_node={mean_lppd:.4f}  "
        f"n_nodes={total_nodes}  "
        f"folds_found={len(records)}/{expected_cv_folds}"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-dirs", nargs="+", required=True,
                        help="Inference output directories, one per CV fold.")
    parser.add_argument("--output",   required=True,
                        help="Destination path for cv_summary.parquet.")
    parser.add_argument("--group-id", required=True,
                        help="Encoded group instance ID (e.g. K~2__stan_file~mixture_simple_contagion).")
    args = parser.parse_args()
    aggregate_cv_folds(args.fold_dirs, args.output, group_id=args.group_id)
