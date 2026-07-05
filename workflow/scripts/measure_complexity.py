"""Snakemake script: collect a sweep's fits, compute kernel complexity per DGP
(rule measure_complexity). 

Writes (with real swept names (x1, x2) restored):
    complexity_per_dgp.parquet     one row per (x1, x2, rep)
    complexity_by_setting.parquet  averaged over reps per (x1, x2)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.analysis import compute_complexity_scores, simple_reference_logit

REG = 1e-6


def load_draws(npz_path, name):
    d = np.load(npz_path)
    if name not in d.files:
        return None
    return np.asarray(d[name], dtype=float)


def build_reference(simple_path, kmax):
    beta = load_draws(simple_path, "draws_beta")
    if beta is None:
        return None
    return simple_reference_logit(float(beta.reshape(-1).mean()), kmax)


def main():
    combos = snakemake.params.combos # list of {file, model, x1, x2, rep}
    axis1_name = snakemake.params.axis1_name
    axis2_name = snakemake.params.axis2_name

    groups: dict[tuple, dict] = {}
    for c in combos:
        key = (c["x1"], c["x2"], c["rep"])
        groups.setdefault(key, {})[c["model"]] = c["file"] # collect np and simple fits

    rows, skipped = [], []
    for (x1, x2, rep), files in groups.items():
        non_simple = [k for k in files if k != "simple"]
        if "simple" not in files or len(non_simple) != 1:
            skipped.append(((x1, x2, rep), f"need exactly one non-simple model + simple, have {sorted(files)}"))
            continue
        np_key = non_simple[0]

        mu_np = load_draws(files[np_key], "draws_mu")
        if mu_np is None:
            skipped.append(((x1, x2, rep), "np fit missing draws_mu"))
            continue
        if mu_np.ndim == 3:
            mu_np = mu_np[:, 0, :]

        ref = build_reference(files["simple"], kmax=mu_np.shape[1])
        if ref is None:
            skipped.append(((x1, x2, rep), "could not build ref (simple fit missing draws_beta)"))
            continue

        try:
            scores = compute_complexity_scores(mu_np, ref, reg=REG)
        except Exception as e:
            skipped.append(((x1, x2, rep), f"complexity failed: {e}"))
            continue

        rows.append({axis1_name: x1, axis2_name: x2, "rep": rep, **scores})

    if not rows:
        print("no usable np/simple pairs found")
        for k, why in skipped[:10]:
            print("  skip", k, why)
        return

    per_dgp = pd.DataFrame(rows)
    per_dgp.to_parquet(snakemake.output[0], index=False)
    print(f"per-DGP: {len(per_dgp)} pairs  ({len(skipped)} skipped)")
    if skipped:
        print(f"  (first few skips: {skipped[:3]})")

    by_setting = (
        per_dgp.groupby([axis1_name, axis2_name], as_index=False)
        .agg(
            complexity_mean=("complexity", "mean"),
            complexity_std=("complexity", "std"),
            n_reps=("complexity", "size"),
            mean_distance=("mean_distance", "mean"),
            erank=("erank", "mean"),
        )
    )
    by_setting.to_parquet(snakemake.output[1], index=False)


if __name__ == "__main__":
    main()
