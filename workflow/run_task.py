"""
Run a single task from a manifest. 
Shouldn't need to be edited when designing experiments.

Usage:
    python run_task.py <manifest.jsonl> <task_id> <results_root>

Dispatches on row["kind"]:
    "inference" -> fit one model, save draws
    "cv"        -> fit + score one CV fold, save one fold record

Skips if the task's output already exists.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# dgp.py (same directory) sets up sys.path and owns ROOT
from dgp import ROOT, get_MN

from src.inference.inference import (
    fit_stan,
    extract_kernel_summary,
    compute_model_comparison,
    get_posterior_draws,
)
from src.inference.cv import run_one_fold


def _atomic_savez(fp: Path, **arrays):
    fp.parent.mkdir(parents=True, exist_ok=True)
    tmp = fp.with_suffix(fp.suffix + ".tmp")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, fp)


def _attach_dgp(df: pd.DataFrame, row: dict) -> pd.DataFrame:
    """Tags every row of df with some experimental conditions."""
    df = df.copy()
    
    for k, v in row["dgp"].items():
        if np.isscalar(v):  # top-level DGP scalars as flat columns
            df[k] = v
    
    # surface nested graph fields as flat columns for easy groupby
    graph = row["dgp"].get("graph", {})
    df["graph_type"] = graph.get("type")
    for k, v in graph.get("params", {}).items():
        if np.isscalar(v):
            df[f"graph_{k}"] = v
    
    df["dgp_key"]      = row["dgp_key"]
    df["task_name"]    = row["task_name"]
    df["model_config"] = row["model"]["config"]
    return df


# inference task (fit one model on full data)
def run_inference_task(M, N, row: dict, out_dir: Path):
    spec   = row["model"]
    result = fit_stan(M, N, ROOT / spec["config"], spec.get("instance_params"))

    draws = {f"draws_{v}": get_posterior_draws(result, v)
             for v in spec.get("save_vars", [])}
    _atomic_savez(out_dir / f"{row['task_name']}.npz", **draws)

    # ksum = extract_kernel_summary(result)
    # if ksum is not None:
    #     _attach_dgp(ksum, row).to_parquet(
    #         out_dir / f"{row['task_name']}_kernel.parquet", index=False)

    # mcomp = compute_model_comparison(result)
    # if mcomp is not None:
    #     _attach_dgp(mcomp, row).to_parquet(
    #         out_dir / f"{row['task_name']}_modelcomp.parquet", index=False)


# CV task (fit + score one fold)
def run_cv_task(M, N, row: dict, out_dir: Path):
    spec = row["model"]
    fr = run_one_fold(
        M, N,
        n_folds=row["cv"]["n_folds"],
        fold_idx=row["cv"]["fold_idx"],
        config_path=ROOT / spec["config"],
        instance_params=spec.get("instance_params"),
    )
    rec = {
        "fold_idx":      fr.fold_idx,
        "cv_score":      fr.cv_score,
        "n_held_out":    fr.n_held_out,
        "mean_log_pred": fr.mean_log_pred,
        "n_folds":       row["cv"]["n_folds"],
        "dgp_label":     row["dgp_label"],
        "dgp_key":       row["dgp_key"],
        "rep":           row["dgp"]["rep"],
        "model_name":    row["model_name"],
        "model_config":  spec["config"],
        "task_name":     row["task_name"],
    }
    pd.DataFrame([rec]).to_parquet(
        out_dir / f"{row['task_name']}_fold.parquet", index=False)


def load_task(manifest: str, task_id: int) -> dict:
    with open(manifest) as f:
        for i, line in enumerate(f):
            if i == task_id:
                return json.loads(line)
    raise IndexError(f"task_id {task_id} >= manifest length")


def main():
    manifest, task_id, results_root = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
    row = load_task(manifest, task_id)

    out_dir = results_root / row["experiment"]
    out_dir.mkdir(parents=True, exist_ok=True)

    sentinel = {
        "inference": out_dir / f"{row['task_name']}.npz",
        "cv":        out_dir / f"{row['task_name']}_fold.parquet",
    }[row["kind"]]
    if sentinel.exists():
        print(f"skip (exists): {row['task_name']}")
        return

    M, N = get_MN(row["dgp"], row["dgp_key"], results_root / "data_cache")
    print(f"run [{row['kind']}]: {row['task_name']}  M={M.shape}")

    if row["kind"] == "inference":
        run_inference_task(M, N, row, out_dir)
    elif row["kind"] == "cv":
        run_cv_task(M, N, row, out_dir)
    else:
        raise ValueError(f"unknown kind: {row['kind']}")


if __name__ == "__main__":
    main()
