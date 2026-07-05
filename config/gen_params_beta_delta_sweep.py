"""Generator for config/params/beta_delta_sweep.tsv. Grid: model x rep x beta2 x delta.
"""

import itertools
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_params_utils import json_col, write_frame, axis_values, resolve_model, model_tag

EXP = "beta_delta_sweep"
WILDCARD_COLUMNS = ["model", "rep", "beta2", "delta"] # swept params listed last

# Nonparametric-EB variant for this experiment (sampled | marginalized)
MODEL_VARIANT = "sampled"
if MODEL_VARIANT == "simple":
    raise ValueError(
        "MODEL_VARIANT == 'simple' is not valid for sweep experiments: "
        "measure_complexity needs one nonparametric fit (sampled/marginalized) "
        "paired with the simple reference."
    )
ACTIVE_MODEL = resolve_model(MODEL_VARIANT)
ACTIVE_TAG = model_tag(MODEL_VARIANT)

# ---- fixed, global (identical every row) -----------------------------------
GRAPH_N = 256
GRAPH_AVG_DEGREE = 18
BETA1 = 0.01
GAMMA = 0.1
RHO = 0.125
TMAX = 1000
EPSILON = 0.9
OMEGA = 0.1
REPS = 10

# ---- swept axes -------------------------------------------------------------
BETA2_AXIS = {"start": 0.01, "step": 0.0025, "count": 17}
DELTA_AXIS = {"start": -0.9, "step": 0.025, "count": 73}

# ---- per-model spec ---------------------------------------------------------
MODELS = {
    ACTIVE_TAG: {
        "model_config": ACTIVE_MODEL["config_path"],
        "instance_params": {
            "K": 1,
            "sampling": {"iter_sampling": 300, "iter_warmup": 900, "chains": 1, "adapt_delta": 0.95},
        },
        "save_vars": ["mu"],
    },
    "simple": {
        "model_config": "config/models/simple.yaml",
        "instance_params": {
            "K": 1,
            "sampling": {"iter_sampling": 300, "iter_warmup": 900, "chains": 1, "adapt_delta": 0.8},
        },
        "save_vars": ["beta", "mu"],
    },
}


def build_frame() -> pd.DataFrame:
    beta2s = axis_values(BETA2_AXIS)
    deltas = axis_values(DELTA_AXIS)
    rows = itertools.product(MODELS.keys(), range(REPS), beta2s, deltas)
    df = pd.DataFrame(rows, columns=["model", "rep", "beta2", "delta"])

    df["graph_n"] = GRAPH_N
    df["graph_avg_degree"] = GRAPH_AVG_DEGREE
    df["beta1"] = BETA1
    df["gamma"] = GAMMA
    df["rho"] = RHO
    df["tmax"] = TMAX
    df["epsilon"] = EPSILON
    df["omega"] = OMEGA

    df["model_config"] = df["model"].map(lambda m: MODELS[m]["model_config"])
    df["instance_params_json"] = df["model"].map(lambda m: json_col(MODELS[m]["instance_params"]))
    df["save_vars_json"] = df["model"].map(lambda m: json_col(MODELS[m]["save_vars"]))
    return df


def main(outdir: Path) -> None:
    write_frame(build_frame(), EXP, WILDCARD_COLUMNS, outdir)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("config/params"))
