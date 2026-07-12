"""Snakemake script: simulate one DGP inline and fit one model on it (rule fit_model)."""

import sys
from pathlib import Path

import numpy as np

WORKFLOW = Path(__file__).resolve().parent.parent
ROOT = WORKFLOW.parent
for p in (ROOT, WORKFLOW):
    if str(p) not in sys.path:
        sys.path.append(str(p))

from dgp import simulate_MN
from src.inference.inference import fit_stan, get_posterior_draws

dgp = dict(snakemake.params.dgp)
spec = dict(snakemake.params.model_spec)
instance_params = dict(spec.get("instance_params") or {})
sampling = dict(instance_params.get("sampling") or {})
sampling.setdefault("seed", int(dgp["rep"]))
instance_params["sampling"] = sampling

M, N = simulate_MN(dgp)

result = fit_stan(M, N, ROOT / spec["config"], instance_params)

draws = {f"draws_{v}": get_posterior_draws(result, v) for v in spec.get("save_vars", [])}
np.savez_compressed(snakemake.output[0], **draws)
