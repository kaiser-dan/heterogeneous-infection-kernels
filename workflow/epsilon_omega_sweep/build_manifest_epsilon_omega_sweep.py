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

EXPERIMENT = "inference_epsilon_omega_sweep" # change
OUT = Path("manifests") / f"{EXPERIMENT}.jsonl"

# Parameters to sweep (edit these lists)
# --------------------------------------
EPSILONS = [round(0.05 * i, 4) for i in range(1, 20)]
OMEGAS   = [round(0.025 * i, 4) for i in range(0, 21)]
REPS     = range(5)

FIXED      = dict(beta1=0.01, beta2=0.04, gamma=0.1, tmax=1000, rho=0.125)   # non-graph params
GRAPH_BASE = dict(n=256, avg_degree=18, delta=0.0)  # fixed sbm params

# Model(s) to fit on each DGP (adding a model = one more entry here)
# ------------------------------------------------------------------
MODELS = {
    "np": {
        "config": "configs/models/nonparametric_eb.yaml",
        "instance_params": {
            "K": 1,
            "sampling": {"iter_sampling": 500, "iter_warmup": 1500,
                         "chains": 1, "adapt_delta": 0.95},
        },
        "save_vars": ["mu"],
    },
    "simple": {
        "config": "configs/models/simple.yaml",
        "instance_params": {
            "K": 1,
            "sampling": {"iter_sampling": 500, "iter_warmup": 1500,
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
    for epsilon, omega, rep in itertools.product(EPSILONS, OMEGAS, REPS):
        # structured DGP form
        dgp = {
            **FIXED,
            "rep":   rep,
            "graph": {"type": "sbm",
                      "params": {**GRAPH_BASE, "epsilon": epsilon}},
            "kernels":    [{"type": "simple", "beta": FIXED["beta1"]},
                           {"type": "simple", "beta": FIXED["beta2"]}],
            "assignment": {"mode": "block_flip", "omega": omega},
        }
        key = dgp_key(dgp)
        
        for mname, mspec in MODELS.items():
            rows.append({
                "experiment": EXPERIMENT,
                "task_name":  f"{mname}_epsilon-{epsilon}_omega-{omega}_rep-{rep}_{key[:8]}",
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
