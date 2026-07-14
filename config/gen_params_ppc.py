"""Generator for config/params/ppc.tsv. Grid: omega x r."""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_params_utils import json_col, write_frame, axis_values

EXP = "ppc"
WILDCARD_COLUMNS = ["omega", "r"]

NREP = 100

# ---- fixed, global (identical every row) -----------------------------------
RHO = 0.125
GAMMA = 0.1
TMAX = 1000

GRAPH_TYPE = "edgelist"
GRAPH_FILENAME = "data/congress.edgelist"

KERNELS = [{"type": "simple", "beta": 0.01}, {"type": "simple", "beta": 0.05}]

ASSIGNMENT_MODE = "block_flip"
ASSIGNMENT_COMMUNITIES_FILE = "data/congress_communities.txt"

NP_CONFIG = "config/models/nonparametric_eb.yaml"
SIMPLE_CONFIG = "config/models/simple.yaml"
SAMPLING = {"iter_sampling": 500, "iter_warmup": 1500}

MASTER_SEED = 20260710 # knob to "re-roll" the experiment
ROLE_OBS_LABEL, ROLE_OBS_DYN, ROLE_OBS_INIT, ROLE_NULL_DYN, ROLE_NULL_INIT = range(5)

# ---- swept axis -------------------------------------------------------------
OMEGA_AXIS = {"start": 0.0, "step": 0.025, "count": 21}

def derive_seed(*path: int) -> int:
    """A deterministic 32-bit seed for one node of the spawn tree under MASTER_SEED.
    Distinct paths -> independent streams."""
    ss = np.random.SeedSequence(MASTER_SEED, spawn_key=tuple(path))
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def build_frame() -> pd.DataFrame:
    omegas = axis_values(OMEGA_AXIS)
    df = pd.DataFrame(itertools.product(omegas, range(NREP)), columns=["omega", "r"])

    df["rho"] = RHO
    df["gamma"] = GAMMA
    df["tmax"] = TMAX

    df["graph_type"] = GRAPH_TYPE
    df["graph_filename"] = GRAPH_FILENAME

    df["kernels_json"] = json_col(KERNELS)

    df["assignment_mode"] = ASSIGNMENT_MODE
    df["assignment_communities_file"] = ASSIGNMENT_COMMUNITIES_FILE

    df["np_config"] = NP_CONFIG
    df["simple_config"] = SIMPLE_CONFIG
    df["sampling_json"] = json_col(SAMPLING)

    df["nrep"] = NREP
    
    # seeds for observed data, constant across omega
    df["seed_label"] = derive_seed(ROLE_OBS_LABEL)
    df["seed_obs"] = derive_seed(ROLE_OBS_DYN)
    df["seed_init"] = derive_seed(ROLE_OBS_INIT)

    # seeds for null data, independent x0 and dynamics per (omega, r)    
    oi = df["omega"].map({om: i for i, om in enumerate(OMEGAS)})
    df["seed_null"] = [derive_seed(ROLE_NULL_DYN, o, r) for o, r in zip(oi, df["r"])]
    df["seed_null_init"] = [derive_seed(ROLE_NULL_INIT, o, r) for o, r in zip(oi, df["r"])]
    
    return df


def main(outdir: Path) -> None:
    write_frame(build_frame(), EXP, WILDCARD_COLUMNS, outdir)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("config/params"))
