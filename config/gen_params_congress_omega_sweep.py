"""Generator for config/params/congress_omega_sweep.tsv. Grid: model x rep x omega."""

import itertools
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_params_utils import json_col, write_frame, axis_values

EXP = "congress_omega_sweep"
WILDCARD_COLUMNS = ["model", "rep", "omega"] # swept param listed last

# ---- fixed, global (identical every row) -----------------------------------
GRAPH_FPATH = "data/congress.edgelist"
COMMUNITIES_FPATH = "data/congress_communities.txt"
BETA1 = 0.01
BETA2 = 0.05
GAMMA = 0.1
RHO = 0.125
TMAX = 1000
REPS = 10

# ---- swept axes -------------------------------------------------------------
OMEGA_AXIS = {"start": 0.0, "step": 0.05, "count": 11}

# ---- per-model spec ---------------------------------------------------------
MODELS = {
    "np": {
        "model_config": "config/models/nonparametric_eb.yaml",
        "instance_params": {
            "stan_file": "src/models/mixture_nonparametric_eb.stan",
            "K": 1,
            "sampling": {"iter_sampling": 500, "iter_warmup": 1500, "chains": 1, "adapt_delta": 0.95},
        },
        "save_vars": ["mu"],
    },
    "simple": {
        "model_config": "config/models/simple.yaml",
        "instance_params": {
            "K": 1,
            "sampling": {"iter_sampling": 500, "iter_warmup": 1500, "chains": 1, "adapt_delta": 0.8},
        },
        "save_vars": ["beta", "mu"],
    },
}


def build_frame() -> pd.DataFrame:
    omegas = axis_values(OMEGA_AXIS)
    rows = itertools.product(MODELS.keys(), range(REPS), omegas)
    df = pd.DataFrame(rows, columns=["model", "rep", "omega"])

    df["graph_fpath"] = GRAPH_FPATH
    df["communities_fpath"] = COMMUNITIES_FPATH
    df["beta1"] = BETA1
    df["beta2"] = BETA2
    df["gamma"] = GAMMA
    df["rho"] = RHO
    df["tmax"] = TMAX

    df["model_config"] = df["model"].map(lambda m: MODELS[m]["model_config"])
    df["instance_params_json"] = df["model"].map(lambda m: json_col(MODELS[m]["instance_params"]))
    df["save_vars_json"] = df["model"].map(lambda m: json_col(MODELS[m]["save_vars"]))
    return df


def main(outdir: Path) -> None:
    write_frame(build_frame(), EXP, WILDCARD_COLUMNS, outdir)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("config/params"))
