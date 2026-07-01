"""
Build a JSONL manifest for inference experiments.
Should be the ONLY file you edit for a new inference sweep. 
Each line is one job (a DGP config + a model to fit on it);
worker (run_task.py) never changes.

Run:  python build_manifest_inference.py   (prints the array range)
"""

import json
import sys
import itertools
from pathlib import Path

import numpy as np

WORKFLOW = Path(__file__).resolve().parent.parent  # workflow/<subdir>/build_*.py -> workflow/
if str(WORKFLOW) not in sys.path:
    sys.path.append(str(WORKFLOW))

from dgp import dgp_key

EXPERIMENT = "inference_beta_delta_sweep" # change
OUT = Path("manifests") / f"{EXPERIMENT}.jsonl"

# Parameters to sweep (edit these lists)
# --------------------------------------
BETA2S   = [round(0.01 + 0.0025*i, 4) for i in range(0, 17)]
DELTAS   = [round(0.025*i, 4) for i in range(-36, 37)]
REPS     = range(5)

FIXED      = dict(beta1=0.01, omega=0.1, gamma=0.1, tmax=1000, rho=0.125)   # non-graph params
GRAPH_BASE = dict(n=256, avg_degree=18, epsilon=0.9)  # fixed sbm params

# Model(s) to fit on each DGP (adding a model = one more entry here)
# ------------------------------------------------------------------
MODELS = {
    "np": {
        "config": "configs/models/nonparametric_eb.yaml",
        "instance_params": {
            "stan_file": "src/models/mixture_nonparametric_eb_vec.stan",
            "K": 1,
            "sampling": {"iter_sampling": 300, "iter_warmup": 900,
                         "chains": 1, "adapt_delta": 0.95},
        },
        "save_vars": ["mu"],
    },
    "simple": {
        "config": "configs/models/simple.yaml",
        "instance_params": {
            "K": 1,
            "sampling": {"iter_sampling": 300, "iter_warmup": 900,
                         "chains": 1, "adapt_delta": 0.8},
        },
        "save_vars": ["beta", "mu"],
    },
}

# DGP defined in here... (edit this)
# ----------------------------------
def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for beta2, delta, rep in itertools.product(BETA2S, DELTAS, REPS):
        # structured DGP form
        dgp = {
            **FIXED,
            "rep":   rep,
            "graph": {"type": "sbm",
                      "params": {**GRAPH_BASE, "delta": delta}},
            "kernels":    [{"type": "simple", "beta": FIXED["beta1"]},
                           {"type": "simple", "beta": beta2}],
            "assignment": {"mode": "block_flip", "omega": FIXED["omega"]},
        }
        key = dgp_key(dgp)
        
        for mname, mspec in MODELS.items():
            rows.append({
                "experiment": EXPERIMENT,
                "task_name":  f"{mname}_beta2-{beta2}_delta-{delta}_rep-{rep}_{key[:8]}",
                "kind":       "inference",
                "dgp_key":    key,
                "dgp":        dgp,
                "model":      mspec,
            })

    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"wrote {len(rows)} tasks to {OUT}")
    print(f"sbatch array range:  --array=0-{len(rows) - 1}")


if __name__ == "__main__":
    main()
