import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def aggregate_cv_folds(fold_dirs: list[str], output_path: str) -> pd.DataFrame:
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
    if df["cv_folds"].nunique() != 1:
        raise ValueError(
            f"Inconsistent cv_folds values across fold dirs: {df['cv_folds'].unique().tolist()}"
        )
    total_nodes = int(df["n_held_out"].sum())
    total_lppd = float(df["cv_score"].sum())

    summary = pd.DataFrame([{
        "cv_folds":              int(df["cv_folds"].iloc[0]),
        "total_cv_lppd":         total_lppd,
        "mean_cv_lppd_per_node": total_lppd / total_nodes if total_nodes > 0 else float("nan"),
        "n_total_nodes":         total_nodes,
    }])
    summary.to_parquet(output_path, index=False)
    print(
        f"CV aggregate: total_cv_lppd={total_lppd:.4f}  "
        f"mean_per_node={summary['mean_cv_lppd_per_node'].iloc[0]:.4f}  "
        f"n_nodes={total_nodes}"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-dirs", nargs="+", required=True,
                        help="Inference output directories for each CV fold")
    parser.add_argument("--output", required=True,
                        help="Path for cv_summary.parquet")
    args = parser.parse_args()
    aggregate_cv_folds(args.fold_dirs, args.output)
