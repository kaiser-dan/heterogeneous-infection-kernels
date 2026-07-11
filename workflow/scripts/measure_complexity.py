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

from src.analysis import (
    compute_complexity_score, 
    simple_reference_logit,
    effective_ledoit_wolf_shrinkage,
)

# fallback REG, used only in degenerate case where no cell yields finite LW shrinkage estimage
REG_FALLBACK = 1e-2


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


def prepare_cell(files):
    """Load and validate one cell's np draws and simple-model reference.

    Returns (mu_np, ref, None) on success, (None, None, reason) if cell can't be scored.
    """
    if "np" not in files or "simple" not in files:
        return None, None, f"need np and simple, have {sorted(files)}"

    mu_np = load_draws(files["np"], "draws_mu")
    if mu_np is None:
        return None, None, "np fit missing draws_mu"
    if mu_np.ndim == 3:
        mu_np = mu_np[:, 0, :]

    ref = build_reference(files["simple"], kmax=mu_np.shape[1])
    if ref is None:
        return None, None, "could not build ref (simple fit missing draws_beta)"

    return mu_np, ref, None


def main():
    combos = snakemake.params.combos # list of {file, model, x1, x2, rep}
    axis1_name = snakemake.params.axis1_name
    axis2_name = snakemake.params.axis2_name

    groups: dict[tuple, dict] = {}
    for c in combos:
        key = (c["x1"], c["x2"], c["rep"])
        groups.setdefault(key, {})[c["model"]] = c["file"] # collect np and simple fits

    # pass 1 to compute effective LW shrinkage per cell
    lw_by_key: dict[tuple, tuple] = {}
    skipped = []
    for key, files in groups.items():
        mu_np, _ref, reason = prepare_cell(files)
        if reason is not None:
            skipped.append((key, reason))
            continue
        try:
            lw = effective_ledoit_wolf_shrinkage(mu_np)
            lw_by_key[key] = (lw["shrinkage"], lw["reg_eff"])
        except Exception as e:
            # keep cell for scoring at global REG_FALLBACK
            lw_by_key[key] = (np.nan, np.nan)
            print(f"  LW shrinkage failed for {key} (cell still scored): {e}")

    if not lw_by_key:
        print("no usable np/simple pairs found")
        for k, why in skipped[:10]:
            print("  skip", k, why)
        return

    reg_eff_all = np.array([reg_eff for (_, reg_eff) in lw_by_key.values()], dtype=float)
    finite = reg_eff_all[np.isfinite(reg_eff_all)]
    global_reg = float(np.median(finite)) if finite.size else REG_FALLBACK
    print(
        f"global REG = median effective LW shrinkage = {global_reg:g} "
        f"(from {finite.size}/{len(lw_by_key)} cells; "
        f"{'data-driven' if finite.size else 'fallback'})"
    )

    # pass 2 to compute complexity per cell
    rows, skipped = [], []
    for key, files in groups.items():
        if key not in lw_by_key:
            continue # already recorded as skipped in pass 1

        mu_np, ref, reason = prepare_cell(files)
        if reason is not None:
            skipped.append((key, reason)) # defensive; deterministic so unexpected
            continue

        try:
            scores = compute_complexity_score(mu_np, ref, reg=global_reg)
        except Exception as e:
            skipped.append((key, f"complexity failed: {e}"))
            continue

        x1, x2, rep = key
        shrinkage, reg_eff = lw_by_key[key]
        rows.append({
            axis1_name: x1, axis2_name: x2, "rep": rep,
            "reg": global_reg,
            "lw_shrinkage": shrinkage, "lw_reg_eff": reg_eff,
            **scores,
        })

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
            trace_p_sigma=("trace_p_sigma", "mean"),
            lw_reg_eff=("lw_reg_eff", "median"),
        )
    )
    by_setting.to_parquet(snakemake.output[1], index=False)


if __name__ == "__main__":
    main()
