"""Snakemake script: fit + score one CV fold on the WeChat counts (rule wechat_cv_task).

Writes one row PER HELD-OUT NODE, in the run_cv_task.py format, so aggregate_cv.py pools it.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.inference.cv import run_one_fold

fixed = dict(snakemake.params.fixed)
K = int(fixed["K"])
fold = int(fixed["fold"])
sampling = json.loads(fixed["sampling_json"])

d = np.load(snakemake.input[0])
fr = run_one_fold(
    d["M"], d["N"],
    n_folds=int(fixed["n_folds"]),
    fold_idx=fold,
    config_path=ROOT / fixed["model_config"],
    instance_params={"K": K, "sampling": {**sampling, "seed": int(fixed["seed"]) + fold}},
)

rec = pd.DataFrame({
    "dgp_label": "wechat",
    "rep":       0,
    "K":         K,
    "fold_idx":  fr.fold_idx,
    "node_idx":  fr.node_idx,
    "log_pred":  fr.log_pred,
})
rec.to_parquet(snakemake.output[0], index=False)
