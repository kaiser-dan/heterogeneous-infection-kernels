"""
Aggregate the parquet files written by run_cv_task's CV tasks (one row per held-out node)
and pools them across folds.

As a Snakemake `script:` (rule aggregate_cv), the fold-file list comes straight from
`snakemake.input`..

Run standalone (globs `*_fold~*.parquet` under the experiment dir, for ad-hoc reruns
outside Snakemake:
    uv run python workflow/scripts/aggregate_cv.py results cv_basic
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_node_lppd(files) -> pd.DataFrame:
    """Concatenate every per-node CV record from the given fold files."""
    files = list(files)
    if not files:
        raise FileNotFoundError("no fold files given")

    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


def cv_curve(nodes: pd.DataFrame) -> pd.DataFrame:
    """One (dgp, rep, K) point per row -> node-weighted mean lppd and per-node SE.

    Pools per-node lppd across folds, so:
      * mean_lppd = sum(log_pred) / V (node-weighted; still correct under unequal
        fold sizes)
      * se_lppd   = sd(log_pred) / sqrt(V) (per-node SE)
    """
    keys = ["dgp_label", "rep", "K"]
    curve = (
        nodes.groupby(keys, dropna=False)
             .agg(mean_lppd=("log_pred", "mean"),   # = sum / V
                  sd_lppd=("log_pred", "std"),      # ddof=1
                  n_nodes=("log_pred", "size"))     # = V
             .reset_index()
    )
    curve["se_lppd"] = curve["sd_lppd"] / np.sqrt(curve["n_nodes"])
    
    return curve


def select_k_1se(curve_one: pd.DataFrame) -> int:
    """1-SE rule for a single (dgp, rep): the most parsimonious K whose mean lppd
    is within one SE of the best K. Higher lppd is better, so the threshold is
    max(mean_lppd) - se(at the argmax K) and we take the smallest K above it.
    """
    c = curve_one.sort_values("K")
    best = c.loc[c["mean_lppd"].idxmax()]
    se = best["se_lppd"]

    if not np.isfinite(se): # degenerate (e.g. a single node) -> fall back to argmax
        return int(best["K"])

    threshold = best["mean_lppd"] - se
    eligible = c[c["mean_lppd"] >= threshold]

    return int(eligible["K"].min())


def selections(curve: pd.DataFrame) -> pd.DataFrame:
    """Apply the 1-SE rule once per (dgp_label, rep)."""
    keys = ["dgp_label", "rep"]
    rows = []
    for (label, rep), g in curve.groupby(keys, dropna=False):
        rows.append({
            "dgp_label":  label,
            "rep":        rep,
            "selected_K": select_k_1se(g),
        })

    return pd.DataFrame(rows)


def recovery_rates(sel: pd.DataFrame, true_k_by_label: dict) -> pd.DataFrame:
    """Fraction of reps where selected_K matches the DGP family's true K."""
    df = sel.copy()
    df["true_K"] = df["dgp_label"].map(true_k_by_label)
    df["correct"] = df["selected_K"] == df["true_K"]
    return (
        df.groupby("dgp_label", as_index=False)
          .agg(true_K=("true_K", "first"),
               recovery_rate=("correct", "mean"),
               n_reps=("correct", "size"))
    )


def run(files, out_dir: Path, true_k_by_label: dict | None = None) -> None:
    """`true_k_by_label` maps dgp_label -> true mixture-component count (config/params/cv_basic.tsv's
    `true_K` column, deduplicated by dgp_label; see gen_params_cv_basic.py)."""
    nodes = load_node_lppd(files)
    curve = cv_curve(nodes)
    sel   = selections(curve)

    out_dir.mkdir(parents=True, exist_ok=True)
    curve.to_parquet(out_dir / "_cv_curve.parquet", index=False)
    sel.to_parquet(out_dir / "_cv_selection.parquet", index=False)

    print(f"pooled {len(nodes)} node-records into {len(curve)} (dgp, rep, K) points")
    print("\nselected K per (case, rep):")
    print(sel.sort_values(["dgp_label", "rep"]).to_string(index=False))

    if true_k_by_label:
        rr = recovery_rates(sel, true_k_by_label)
        print("\nrecovery rate by case (selected_K == true K):")
        print(rr.to_string(index=False))
    else:
        print("\n(no true_k_by_label given, skipping recovery-rate summary)")


def main():
    """Standalone CLI entrypoint for ad-hoc reruns outside Snakemake."""
    results_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    experiment   = sys.argv[2] if len(sys.argv) > 2 else "cv_basic"

    out_dir = results_root / experiment
    files = sorted(out_dir.glob("*_fold~*.parquet"))

    root = Path(__file__).resolve().parent.parent.parent
    cv_frame = pd.read_csv(root / "config" / "params" / f"{experiment}.tsv", sep="\t")
    true_k_by_label = cv_frame.drop_duplicates("dgp_label").set_index("dgp_label")["true_K"].to_dict()
    run(files, out_dir, true_k_by_label)


try:
    snakemake
except NameError:
    if __name__ == "__main__":
        main()
else:
    run(
        snakemake.input,
        Path(snakemake.output[0]).parent,
        dict(snakemake.params.true_k_by_label),
    )
