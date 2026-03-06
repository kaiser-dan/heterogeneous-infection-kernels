#!/usr/bin/env python3
"""
Score OO fit quality for a sweep by comparing fitted curves to empirical P(adopt|k).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def load_empirical_curve(
    repo_root: Path,
    sim_id: int,
    method: str,
    kmax: int,
    drop_zero_bin: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    base = repo_root / "data" / "processed" / "oo" / f"sim_{sim_id}_backbone_{method}"
    m_df = pd.read_parquet(base / "M.parquet")
    n_df = pd.read_parquet(base / "N.parquet")
    cols = sorted([c for c in m_df.columns if str(c).isdigit()], key=lambda x: int(str(x)))

    M = m_df[cols].sum(axis=0).to_numpy(dtype=float)
    N = n_df[cols].sum(axis=0).to_numpy(dtype=float)
    if drop_zero_bin:
        if len(M) <= 1:
            raise ValueError("Cannot drop zero-exposure bin: empirical curve has <=1 bin.")
        M = M[1:]
        N = N[1:]
    if len(M) > kmax:
        M, N = M[:kmax], N[:kmax]
    elif len(M) < kmax:
        pad = kmax - len(M)
        M = np.pad(M, (0, pad))
        N = np.pad(N, (0, pad))
    return M, N


def _parse_component_means(summary_df: pd.DataFrame, kmax: int) -> dict[int, np.ndarray]:
    pat = re.compile(r"^mu\[(\d+),(\d+)\]$")
    comp_vals: dict[int, np.ndarray] = {}
    for idx, val in summary_df["Mean"].items():
        m = pat.match(str(idx))
        if not m:
            continue
        comp = int(m.group(1))
        nu = int(m.group(2))
        if comp not in comp_vals:
            comp_vals[comp] = np.full(kmax, np.nan, dtype=float)
        if 1 <= nu <= kmax:
            comp_vals[comp][nu - 1] = float(val)
    return comp_vals


def _extract_pi(summary_df: pd.DataFrame, n_components: int) -> np.ndarray:
    pi = np.zeros(n_components, dtype=float)
    for c in range(1, n_components + 1):
        key = f"pi[{c}]"
        if key in summary_df.index:
            pi[c - 1] = float(summary_df.loc[key, "Mean"])
    if np.allclose(pi.sum(), 0.0):
        pi[:] = 1.0 / n_components
    else:
        pi /= pi.sum()
    return pi


def fitted_curve(run_dir: Path, kmax: int) -> np.ndarray | None:
    summary_path = run_dir / "inference_summary.parquet"
    if not summary_path.exists():
        return None
    df = pd.read_parquet(summary_path)

    # SC/CC in this project expose c[nu].
    c_vals = np.full(kmax, np.nan, dtype=float)
    found_c = False
    for nu in range(1, kmax + 1):
        key = f"c[{nu}]"
        if key in df.index:
            c_vals[nu - 1] = float(df.loc[key, "Mean"])
            found_c = True
    if found_c:
        return c_vals

    # Preferred path for NP monotonic in this repo.
    km_path = run_dir / "kernel_mode_summary.parquet"
    if km_path.exists():
        kdf = pd.read_parquet(km_path)
        if not kdf.empty:
            n_components = int(kdf["component"].max())
            pi = _extract_pi(df, n_components)
            pred = np.full(kmax, np.nan, dtype=float)
            for nu in range(1, kmax + 1):
                sub = kdf[kdf["nu"] == nu]
                if sub.empty:
                    continue
                val = 0.0
                for c in range(1, n_components + 1):
                    csub = sub[sub["component"] == c]
                    if csub.empty:
                        continue
                    val += pi[c - 1] * float(csub["mean"].iloc[0])
                pred[nu - 1] = val
            return pred

    # Fallback: aggregate mu[component, nu] with pi weights.
    comp_means = _parse_component_means(df, kmax)
    if comp_means:
        comps = sorted(comp_means.keys())
        pi = _extract_pi(df, len(comps))
        pred = np.full(kmax, np.nan, dtype=float)
        for i, c in enumerate(comps):
            vals = comp_means[c]
            pred = np.where(np.isnan(pred), 0.0, pred) + pi[i] * np.where(np.isnan(vals), 0.0, vals)
        return pred

    return None


def score_run(run_dir: Path, M: np.ndarray, N: np.ndarray) -> dict[str, float] | None:
    kmax = len(M)
    pred = fitted_curve(run_dir, kmax=kmax)
    if pred is None:
        return None
    total = M + N
    with np.errstate(divide="ignore", invalid="ignore"):
        obs = np.where(total > 0, M / total, np.nan)
    valid = (total > 0) & np.isfinite(obs) & np.isfinite(pred)
    if not np.any(valid):
        return None
    weighted_mae = float(np.sum(total[valid] * np.abs(pred[valid] - obs[valid])) / np.sum(total[valid]))
    peak_underfit = float(np.max(obs[valid] - pred[valid]))
    mean_ratio = float(np.mean(pred[valid] / obs[valid]))
    tail_mask = valid & (np.arange(1, len(obs) + 1) >= 10)
    if np.any(tail_mask):
        tail_mae_k10plus = float(np.mean(np.abs(pred[tail_mask] - obs[tail_mask])))
    else:
        tail_mae_k10plus = np.nan
    peak_k = int(np.nanargmax(obs) + 1) if np.any(np.isfinite(obs)) else -1
    peak_obs = float(np.nanmax(obs)) if np.any(np.isfinite(obs)) else np.nan
    peak_pred = float(pred[peak_k - 1]) if 1 <= peak_k <= len(pred) and np.isfinite(pred[peak_k - 1]) else np.nan
    return {
        "weighted_mae": weighted_mae,
        "tail_mae_k10plus": tail_mae_k10plus,
        "peak_underfit": peak_underfit,
        "mean_model_to_empirical": mean_ratio,
        "peak_k": peak_k,
        "peak_obs": peak_obs,
        "peak_pred": peak_pred,
        "k1_pred": float(pred[0]) if np.isfinite(pred[0]) else np.nan,
        "k10_pred": float(pred[9]) if len(pred) >= 10 and np.isfinite(pred[9]) else np.nan,
        "k15_pred": float(pred[14]) if len(pred) >= 15 and np.isfinite(pred[14]) else np.nan,
        "k20_pred": float(pred[19]) if len(pred) >= 20 and np.isfinite(pred[19]) else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", required=True, help="Sweep directory name under results/")
    parser.add_argument("--sim-id", type=int, default=165)
    parser.add_argument("--backbone-method", default="disparity")
    parser.add_argument("--kmax", type=int, default=20)
    parser.add_argument(
        "--drop-zero-bin",
        action="store_true",
        help="Drop empirical exposure bin 0 (k=0) before scoring.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional output CSV path. Defaults to results/<sweep>/oo_fit_gap_summary.csv",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    run_base = repo_root / "results" / args.sweep / "oo" / f"sim_{args.sim_id}_backbone_{args.backbone_method}_exposures"
    if not run_base.exists():
        raise FileNotFoundError(f"Run directory not found: {run_base}")

    M, N = load_empirical_curve(
        repo_root,
        args.sim_id,
        args.backbone_method,
        args.kmax,
        drop_zero_bin=args.drop_zero_bin,
    )
    rows = []
    for summary_path in run_base.rglob("inference_summary.parquet"):
        run_dir = summary_path.parent
        score = score_run(run_dir, M, N)
        if score is None:
            continue
        rel = run_dir.relative_to(run_base)
        row = {"instance_id": str(rel)}
        for part in rel.parts:
            if "~" in part:
                key, value = part.split("~", 1)
                row[key] = value
        row.update(score)
        rows.append(row)

    if not rows:
        print("No scored runs found.")
        return

    out_df = pd.DataFrame(rows).sort_values("weighted_mae")
    out_path = Path(args.output_csv) if args.output_csv else (repo_root / "results" / args.sweep / "oo_fit_gap_summary.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} scored runs to {out_path}")
    print("\nTop 10 by weighted_mae:")
    print(out_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
