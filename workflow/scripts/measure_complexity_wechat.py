"""Snakemake script: complexity of every WeChat kernel mode against the closest simple contagion
with a baseline, 1 - (1 - beta0)(1 - beta)^nu (rule wechat_complexity).

Users who view a page share it with probability ~0.4 at nu = 1, so the reference needs beta0.
REG is the median effective Ledoit-Wolf shrinkage over the modes, as in measure_complexity_1d.py.
Writes complexity.parquet: one row per (K, component).
"""

import re
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.analysis import (
    compute_complexity_score,
    offset_reference_logit,
    effective_ledoit_wolf_shrinkage,
)

GRID = np.linspace(-8, 4, 13)  # logit beta0, logit beta starting points


def closest_offset(mu, reg):
    """Minimise complexity over (beta0, beta), from the best point of a coarse grid."""
    kmax = mu.shape[1]
    score = lambda x: compute_complexity_score(mu, offset_reference_logit(*expit(x), kmax), reg=reg)["complexity"]
    x0 = min(product(GRID, GRID), key=score)
    x = minimize(score, x0, method="Nelder-Mead").x
    return expit(x), compute_complexity_score(mu, offset_reference_logit(*expit(x), kmax), reg=reg)


modes = []
for f in snakemake.input:
    K = int(re.search(r"K~(\d+)", f).group(1))
    mu = np.load(f)["draws_mu"]
    modes += [(K, k, mu[:, k, :]) for k in range(K)]

reg = float(np.median([effective_ledoit_wolf_shrinkage(mu)["reg_eff"] for _, _, mu in modes]))
print(f"REG = median effective LW shrinkage = {reg:g}")

rows = []
for K, k, mu in modes:
    (beta0, beta), scores = closest_offset(mu, reg)
    rows.append({"K": K, "component": k + 1, "beta0": beta0, "beta": beta, "reg": reg, **scores})

df = pd.DataFrame(rows)
df.to_parquet(snakemake.output[0], index=False)
print(df.to_string(index=False))
