"""Snakemake script: simulate one CV-sweep DGP, then fit and score one fold (rule run_cv_sweep_task).

Writes one row per held-out node: cell's axis values, dgp, true_K, K, rep, fold_idx,
node_idx, log_pred (held-out LPPD), node_M / node_N (that node's total infection and
non-infection events, so per-rep exposure totals can be recovered), and n_divergent
(divergent transitions in this fold's fit, repeated on every row).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORKFLOW = Path(__file__).resolve().parent.parent
ROOT = WORKFLOW.parent
for p in (ROOT, WORKFLOW):
    if str(p) not in sys.path:
        sys.path.append(str(p))

from dgp import simulate_MN
from src.inference.cv import _cv_fold_mask, compute_held_out_loglik
from src.inference.inference import fit_stan

fixed = dict(snakemake.params.fixed)
dgp = dict(snakemake.params.dgp)
axes = list(snakemake.params.axes)

dgp_label = fixed["dgp"]
K = int(fixed["K"])
rep = int(fixed["rep"])
fold = int(fixed["fold"])
n_folds = int(fixed["n_folds"])
sampling = json.loads(fixed["sampling_json"])

M, N = simulate_MN(dgp)
M = np.asarray(M, dtype=float)
N = np.asarray(N, dtype=float)
V, kmax = M.shape

if dgp_label == "oneMatched":
    # check: matched kernel was fitted on this (cell, rep)'s graph, so it must have exactly kmax levels
    n_levels = np.loadtxt(dgp["kernels"][0]["values_file"]).size
    if n_levels != kmax:
        raise RuntimeError(
            f"matched kernel has {n_levels} exposure levels but this graph has kmax={kmax}; "
            "the stage-1 fit was made on a different graph realization."
        )

held = _cv_fold_mask(V, n_folds, fold)
instance_params = {"K": K, "sampling": {**sampling, "seed": rep * n_folds + fold}}
result = fit_stan(M[~held], N[~held], ROOT / fixed["model_config"], instance_params)
scores = compute_held_out_loglik(M[held], N[held], result.fit)
n_divergent = int(np.asarray(result.fit.method_variables()["divergent__"]).sum())

rec = pd.DataFrame({
    **{a: fixed[a] for a in axes},
    "dgp": dgp_label,
    "true_K": int(fixed["true_K"]),
    "rep": rep,
    "K": K,
    "fold_idx": fold,
    "node_idx": np.flatnonzero(held),
    "log_pred": scores["log_pred"],
    "node_M": M[held].sum(axis=1),
    "node_N": N[held].sum(axis=1),
    "n_divergent": n_divergent,
})
rec.to_parquet(snakemake.output[0], index=False)
