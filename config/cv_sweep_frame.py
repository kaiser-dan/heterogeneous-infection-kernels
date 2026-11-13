"""Shared builder for the gen_params_cv_<exp>.py generators.

Grid: <cell axes> x DGP x K x rep x fold. Each cell has two DGPs, paired by rep
(both DGPs of a (cell, rep) share the same graph, initial state and seeds):

  twoSimple  : two (K=2) simple kernels (beta1, beta2) assigned by block_flip(omega).
  oneMatched : one (K=1) nonparametric kernel, equal to the posterior-mean K=1 kernel 
               mode fitted to the twoSimple data of the same (cell, rep).

The matched kernel is fitted in workflow/rules/cv_sweep.smk (stage 1).
"""

from __future__ import annotations

import itertools

import pandas as pd

from gen_params_utils import json_col

DGPS = {"twoSimple": 2, "oneMatched": 1}  # DGP -> true K
K_VALUES = [1, 2, 3]
REPS = 20
N_FOLDS = 5

# fixed, same as cv_basic
GRAPH_N = 256
GRAPH_AVG_DEGREE = 18
GAMMA = 0.1
RHO = 0.25
TMAX = 1000

# stage 1 (matched-kernel fit): same spec as the "np" model of epsilon_omega_sweep / beta_delta_sweep
STAGE1_MODEL_SPEC = {
    "config": "config/models/nonparametric_eb.yaml",
    "instance_params": {
        "stan_file": "src/models/mixture_nonparametric_eb.stan",
        "K": 1,
        "sampling": {"iter_sampling": 300, "iter_warmup": 900, "chains": 1, "adapt_delta": 0.95},
    },
    "save_vars": ["mu"],
}

# stage 2 (CV fits): same model and sampler settings as cv_basic
CV_MODEL_CONFIG = "config/models/nonparametric_eb.yaml"
CV_SAMPLING = {"iter_sampling": 300, "iter_warmup": 900, "chains": 1, "adapt_delta": 0.95}

STRUCTURAL_COLUMNS = ["epsilon", "omega", "beta1", "beta2", "delta"]

def cell_id(cell: dict, axes: list[str]) -> str:
    """Filename-safe cell label, e.g. 'epsilon~0.05_omega~0.0'."""
    return "_".join(f"{a}~{cell[a]}" for a in axes)


def build_cv_sweep_frame(cells: list[dict], axes: list[str]) -> pd.DataFrame:
    """One row per (cell, dgp, K, rep, fold). Each cell dict must give every STRUCTURAL_COLUMNS value."""
    for cell in cells:
        missing = [c for c in STRUCTURAL_COLUMNS if c not in cell]
        if missing:
            raise ValueError(f"cell {cell} is missing {missing}")

    rows = []
    for cell, dgp, K, rep, fold in itertools.product(cells, DGPS, K_VALUES, range(REPS), range(N_FOLDS)):
        rows.append({
            **{c: cell[c] for c in STRUCTURAL_COLUMNS},
            "dgp": dgp,
            "K": K,
            "rep": rep,
            "fold": fold,
            "cell": cell_id(cell, axes),
            "true_K": DGPS[dgp],
        })
    df = pd.DataFrame(rows)

    df["n_folds"] = N_FOLDS
    df["graph_n"] = GRAPH_N
    df["graph_avg_degree"] = GRAPH_AVG_DEGREE
    df["gamma"] = GAMMA
    df["rho"] = RHO
    df["tmax"] = TMAX
    df["model_config"] = CV_MODEL_CONFIG
    df["sampling_json"] = json_col(CV_SAMPLING)
    df["stage1_model_spec_json"] = json_col(STAGE1_MODEL_SPEC)
    return df
