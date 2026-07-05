"""Generator for config/params/cv.tsv. Grid: dgp_label x K x rep x fold."""

import itertools
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_params_utils import json_col, write_frame

EXP = "cv_basic"
WILDCARD_COLUMNS = ["dgp_label", "K", "rep", "fold"]

# ---- fixed, global (identical every row) -----------------------------------
GAMMA = 0.1
RHO = 0.25
TMAX = 1000
N_FOLDS = 5
REPS = 20

GRAPH_TYPE = "sbm"
GRAPH_N = 256
GRAPH_AVG_DEGREE = 18
GRAPH_EPSILON = 0.9
GRAPH_DELTA = 0.0

MODEL_CONFIG = "config/models/nonparametric_eb.yaml"
STAN_FILE = "src/models/mixture_nonparametric_eb.stan"
SAMPLING = {"iter_sampling": 300, "iter_warmup": 900, "chains": 1, "adapt_delta": 0.95}

K_VALUES = [1, 2, 3, 4]

# ---- per-dgp_label spec ----------------------------------------------------
DGP_FAMILIES = {
    "twoSimpleOmega0.1": {
        "kernels": [{"type": "simple", "beta": 0.01}, {"type": "simple", "beta": 0.04}],
        "assignment": {"mode": "block_flip", "omega": 0.1},
    },
    "oneComplex": {
        "kernels": [{"type": "nonparametric", "values_file": "data/np_complex_kernel.txt"}],
        "assignment": {},
    },
    "twoComplexOmega0.1": {
        "kernels": [{"type": "power", "lam": 0.005, "alpha": 1.8}, {"type": "power", "lam": 0.15, "alpha": 0.4}],
        "assignment": {"mode": "block_flip", "omega": 0.1},
    },
    "oneSimple": {
        "kernels": [{"type": "simple", "beta": 0.035}],
        "assignment": {},
    },
}


def build_frame() -> pd.DataFrame:
    rows = itertools.product(DGP_FAMILIES.keys(), K_VALUES, range(REPS), range(N_FOLDS))
    df = pd.DataFrame(rows, columns=["dgp_label", "K", "rep", "fold"])

    df["gamma"] = GAMMA
    df["rho"] = RHO
    df["tmax"] = TMAX
    df["n_folds"] = N_FOLDS

    df["graph_type"] = GRAPH_TYPE
    df["graph_n"] = GRAPH_N
    df["graph_avg_degree"] = GRAPH_AVG_DEGREE
    df["graph_epsilon"] = GRAPH_EPSILON
    df["graph_delta"] = GRAPH_DELTA

    df["model_config"] = MODEL_CONFIG
    df["stan_file"] = STAN_FILE
    df["sampling_json"] = json_col(SAMPLING)

    df["kernels_json"] = df["dgp_label"].map(lambda lb: json_col(DGP_FAMILIES[lb]["kernels"]))
    df["assignment_json"] = df["dgp_label"].map(lambda lb: json_col(DGP_FAMILIES[lb]["assignment"]))
    df["true_K"] = df["dgp_label"].map(lambda lb: len(DGP_FAMILIES[lb]["kernels"]))
    return df


def main(outdir: Path) -> None:
    write_frame(build_frame(), EXP, WILDCARD_COLUMNS, outdir)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("config/params"))
