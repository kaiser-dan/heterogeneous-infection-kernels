"""Snakemake script: simulate one CV DGP inline, then fit + score one fold
(rule run_cv_task).

Writes one row PER HELD-OUT NODE; columns: dgp_label, rep, K,
log_pred, plus node_idx/fold_idx for provenance.
"""

import json
import sys
from pathlib import Path

import pandas as pd

WORKFLOW = Path(__file__).resolve().parent.parent
ROOT = WORKFLOW.parent
for p in (ROOT, WORKFLOW):
    if str(p) not in sys.path:
        sys.path.append(str(p))

from dgp import simulate_MN
from src.inference.cv import run_one_fold

fixed = dict(snakemake.params.fixed)

K = int(fixed["K"])
rep = int(fixed["rep"])
fold = int(fixed["fold"])
n_folds = int(fixed["n_folds"])
sampling = json.loads(fixed["sampling_json"])

instance_params = {
    "K": K,
    "sampling": {**sampling, "seed": rep * n_folds + fold},  # deterministic given (rep, fold)
}

dgp = dict(snakemake.params.dgp)
M, N = simulate_MN(dgp)

fr = run_one_fold(
    M, N,
    n_folds=n_folds,
    fold_idx=fold,
    config_path=ROOT / fixed["model_config"],
    instance_params=instance_params,
)

# one row per held-out node
rec = pd.DataFrame({
    "dgp_label": fixed["dgp_label"],
    "rep":       rep,
    "K":         K,
    "fold_idx":  fr.fold_idx,
    "node_idx":  fr.node_idx,
    "log_pred":  fr.log_pred,
})
rec.to_parquet(snakemake.output[0], index=False)
