"""
Build a JSONL manifest for the CV model-selection experiments.
Should be the ONLY file you edit for a new CV sweep.
Each line is one (dgp, rep, candidate model, fold)-tuple; 
aggregator sums folds back into one CV score per (dgp, rep, candidate).

Run:  uv run python workflow/build_manifest_cv.py   (prints the array range)
"""

import json
from pathlib import Path

from dgp import dgp_key

EXPERIMENT = "cv_complex_vs_simple" # change
OUT = Path("manifests") / f"{EXPERIMENT}.jsonl"

N_FOLDS = 5
REPS    = range(5)

# Non-graph DGP params
FIXED = dict(gamma=0.1, rho=0.125, tmax=1000)

# Fixed graph for the whole sweep
GRAPH = {"type": "sbm",
         "params": {"n": 256, "avg_degree": 18, "epsilon": 0.9, "delta": 0.0}}

# Ground-truth DGP families; `label` is bookkeeping only
# ------------------------------------------------------
DGP_FAMILIES = [
    {
        "label":      "two_simple_omega0.1",
        "kernels":    [{"type": "simple", "beta": 0.01},
                       {"type": "simple", "beta": 0.05}],
        "assignment": {"mode": "block_flip", "omega": 0.1},
    },
    {
        "label":      "one_complex",
        "kernels":    [{"type": "hill", "p": 0.7, "h": 8.0, "tau": 12}],
        "assignment": {}, # single kernel -> unused
    },
]

# Candidate models to compare via CV
# ----------------------------------
_SAMPLING = {"iter_sampling": 500, "iter_warmup": 1500, "chains": 1}

CANDIDATES = {
    "np_K1": {
        "config": "configs/models/nonparametric_eb.yaml",
        "instance_params": {"K": 1, "sampling": {**_SAMPLING, "adapt_delta": 0.95}},
    },
    "np_K2": {
        "config": "configs/models/nonparametric_eb.yaml",
        "instance_params": {"K": 2, "sampling": {**_SAMPLING, "adapt_delta": 0.95}},
    },
}


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for fam in DGP_FAMILIES:
        for rep in REPS:
            # structured DGP form
            dgp = {
                **FIXED,
                "rep":        rep,
                "graph":      GRAPH,
                "kernels":    fam["kernels"],
                "assignment": fam["assignment"],
            }
            key = dgp_key(dgp)
            
            for mname, mspec in CANDIDATES.items():
                for fold in range(N_FOLDS):
                    rows.append({
                        "experiment": EXPERIMENT,
                        "kind":       "cv",
                        "task_name":  f"{fam['label']}__{mname}__rep-{rep}"
                                      f"__fold-{fold}_{key[:8]}",
                        "dgp_label":  fam["label"],
                        "dgp_key":    key,
                        "dgp":        dgp,
                        "model_name": mname,
                        "model":      mspec,
                        "cv":         {"n_folds": N_FOLDS, "fold_idx": fold},
                    })

    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"wrote {len(rows)} tasks to {OUT}")
    print(f"sbatch array range:  --array=0-{len(rows) - 1}")


if __name__ == "__main__":
    main()
