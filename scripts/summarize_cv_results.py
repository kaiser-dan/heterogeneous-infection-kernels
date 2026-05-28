"""
Walk CV sweep results and write a single summary parquet for downstream plots.

Output: results/cv_summary.parquet -- tidy union schema with a `kind` discriminator.

- kind="cv_lppd"          -> one row per (sweep_name, dataset, k_fit):
                             mean / SE of held-out lppd-per-node across folds.
- kind="kernel_recovery"  -> one row per (sweep_name, dataset, k_fit, component, nu):
                             kernel mode summary (median + CI) from the fold with
                             maximum nu=1 component spread (the "best" fold).

Usage:
    uv run python scripts/summarize_cv_results.py [--output PATH]
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).parent.parent / "results"

# (sweep_name, dataset, true_k) -- the producer authoritatively lists which
# sweeps belong in the summary. Plot-side display metadata (betas, colors)
# stays in the plot script.
SWEEPS = [
    ("cv_model_sel_k1_gt_20260501", "cv_k1_ground_truth", 1),
    ("cv_model_sel_k2_gt_20260501", "cv_k2_ground_truth", 2),
]

K_FITS = [1, 2, 3, 4]
NU_MAX = 15


def collect_cv_lppd(sweep_name: str, dataset: str, true_k: int) -> list[dict]:
    base = RESULTS / sweep_name / "synthetic" / dataset
    rows = []
    for k_dir in sorted(base.glob("K~*"), key=lambda p: int(p.name.split("~")[1])):
        k = int(k_dir.name.split("~")[1])
        fold_means = [
            pd.read_parquet(p)["mean_log_pred"].iloc[0]
            for p in sorted(k_dir.glob("cv_fold~*/cv_held_out_loglik.parquet"))
        ]
        if not fold_means:
            continue
        arr = np.array(fold_means)
        n = len(arr)
        rows.append({
            "kind": "cv_lppd",
            "sweep_name": sweep_name,
            "dataset": dataset,
            "true_k": true_k,
            "k_fit": k,
            "mean_cv_lppd_per_node": float(arr.mean()),
            "se_cv_lppd_per_node": float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            "n_folds": n,
        })
    return rows


def select_best_fold(sweep_name: str, dataset: str, k_fit: int) -> tuple[str | None, int | None]:
    """Return (path, fold_id) of the fold maximising nu=1 component spread."""
    pattern = str(
        RESULTS / sweep_name / "synthetic" / dataset / f"K~{k_fit}" / "cv_fold~*"
        / "kernel_mode_summary.parquet"
    )
    best_path, best_spread, best_fold = None, -1.0, None
    for p in sorted(glob.glob(pattern)):
        df = pd.read_parquet(p)
        nu1 = df[df.nu == 1].groupby("component")["median"].mean()
        spread = float(nu1.max() - nu1.min()) if len(nu1) > 1 else 0.0
        if spread > best_spread:
            best_spread = spread
            best_path = p
            best_fold = int(Path(p).parent.name.split("~")[1])
    return best_path, best_fold


def collect_kernel_recovery(sweep_name: str, dataset: str, true_k: int) -> list[dict]:
    rows = []
    for k_fit in K_FITS:
        path, fold_id = select_best_fold(sweep_name, dataset, k_fit)
        if path is None:
            continue
        df = pd.read_parquet(path)
        df = df[df.nu <= NU_MAX]
        for _, r in df.iterrows():
            rows.append({
                "kind": "kernel_recovery",
                "sweep_name": sweep_name,
                "dataset": dataset,
                "true_k": true_k,
                "k_fit": k_fit,
                "best_fold": fold_id,
                "component": int(r["component"]),
                "nu": int(r["nu"]),
                "median": float(r["median"]),
                "ci_low": float(r["ci_low"]),
                "ci_high": float(r["ci_high"]),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(RESULTS / "cv_summary.parquet"),
        help="Destination parquet path (default: results/cv_summary.parquet).",
    )
    args = parser.parse_args()

    rows: list[dict] = []
    for sweep_name, dataset, true_k in SWEEPS:
        rows.extend(collect_cv_lppd(sweep_name, dataset, true_k))
        rows.extend(collect_kernel_recovery(sweep_name, dataset, true_k))

    if not rows:
        raise SystemExit(
            "No CV results found under results/. Run sweeps and `just sync` first."
        )

    df = pd.DataFrame(rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    print(f"Wrote {len(df)} rows to {out}")
    print(df.groupby(["kind", "dataset"]).size().to_string())


if __name__ == "__main__":
    main()
