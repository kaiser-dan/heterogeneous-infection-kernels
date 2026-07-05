"""Generator for config/params/ppc.tsv. Only thing that varies row-to-row is the null-replicate index `r`."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_params_utils import json_col, write_frame, resolve_model

EXP = "ppc"
WILDCARD_COLUMNS = ["r"]

NREP = 100

RHO = 0.125
GAMMA = 0.1
TMAX = 1000

GRAPH_TYPE = "edgelist"
GRAPH_FILENAME = "data/congress.edgelist"

KERNELS = [{"type": "simple", "beta": 0.01}, {"type": "simple", "beta": 0.05}]

ASSIGNMENT_MODE = "block_flip"
ASSIGNMENT_OMEGA = 0.05
ASSIGNMENT_COMMUNITIES_FILE = "data/congress_communities.txt"

# Nonparametric-EB variant for this experiment (sampled | marginalized)
MODEL_VARIANT = "marginalized"
if MODEL_VARIANT == "simple":
    raise ValueError("MODEL_VARIANT == 'simple' is not valid for the PPC experiment.")
ACTIVE_MODEL = resolve_model(MODEL_VARIANT)

NP_CONFIG = ACTIVE_MODEL["config_path"]
SIMPLE_CONFIG = "config/models/simple.yaml"
SAMPLING = {"iter_sampling": 500, "iter_warmup": 1500}

REG = 1.0e-6
SEED_LABEL = 11
SEED_INIT = 17
SEED_OBS = 13
SEED_NULL_BASE = 1_000_000   # null dynamics seeds live at BASE + r, disjoint from the fixed seeds above

def build_frame() -> pd.DataFrame:
    df = pd.DataFrame({"r": range(NREP)})

    df["rho"] = RHO
    df["gamma"] = GAMMA
    df["tmax"] = TMAX

    df["graph_type"] = GRAPH_TYPE
    df["graph_filename"] = GRAPH_FILENAME

    df["kernels_json"] = json_col(KERNELS)

    df["assignment_mode"] = ASSIGNMENT_MODE
    df["assignment_omega"] = ASSIGNMENT_OMEGA
    df["assignment_communities_file"] = ASSIGNMENT_COMMUNITIES_FILE

    df["np_config"] = NP_CONFIG
    df["simple_config"] = SIMPLE_CONFIG
    df["sampling_json"] = json_col(SAMPLING)

    df["nrep"] = NREP
    df["reg"] = REG
    df["seed_label"] = SEED_LABEL
    df["seed_init"] = SEED_INIT
    df["seed_obs"] = SEED_OBS
    df["seed_null_base"] = SEED_NULL_BASE
    return df


def main(outdir: Path) -> None:
    write_frame(build_frame(), EXP, WILDCARD_COLUMNS, outdir)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("config/params"))
