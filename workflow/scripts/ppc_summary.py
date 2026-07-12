"""Snakemake script: collect observed + null PPC fits, compute p-value (rule ppc_summary)."""

import sys
from pathlib import Path

import numpy as np
import json

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.analysis import ppc_complexity

fixed = dict(snakemake.params.fixed)
reg = float(fixed["reg"])

obs = np.load(snakemake.input.observed)
mus, beta_hat = obs["mus"], float(obs["beta_hat"])

mu_rep_draws = [np.load(f)["mu_null"] for f in snakemake.input.nulls]

res = ppc_complexity(mus, mu_rep_draws, beta=beta_hat, reg=reg, plus_one=True)

summary = {
    "p_value":  res["p_value"],
    "d_obs":    res["d_obs"],
    "n_null":   res["n_null"],
    "n_ge":     res["n_ge"],
    "R_requested": int(fixed["nrep"]),
    "beta_hat": beta_hat,
    "reg":      reg,
}
Path(snakemake.output.result_json).write_text(json.dumps(summary, indent=2))
np.savez_compressed(snakemake.output.null_dist, d_null=res["d_null"], d_obs=np.array(res["d_obs"]))

print("\nPPC result:")
print(json.dumps(summary, indent=2))
