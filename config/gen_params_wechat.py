"""Generator for config/params/wechat.tsv. Grid: K x fold."""

import itertools
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_params_utils import json_col, write_frame

EXP = "wechat"
WILDCARD_COLUMNS = ["K", "fold"]

# ---- fixed, global (identical every row) -----------------------------------
WINDOWS = ["201708010810", "201708110820", "201708210830"]  # WMdata raw logs, ten days each
KMAX = 20
N_FOLDS = 5
SEED = 20261017

MODEL_CONFIG = "config/models/nonparametric_eb_cells.yaml"
SAMPLING = {"iter_sampling": 1000, "iter_warmup": 1000, "chains": 4, "adapt_delta": 0.95}

SBM_INITS = 20     # minimize_blockmodel_dl restarts; smallest description length kept
N_PERM = 2000      # label permutations for the community null

K_VALUES = [1, 2]


def build_frame() -> pd.DataFrame:
    df = pd.DataFrame(itertools.product(K_VALUES, range(N_FOLDS)), columns=["K", "fold"])

    df["windows_json"] = json_col(WINDOWS)
    df["kmax"] = KMAX
    df["n_folds"] = N_FOLDS
    df["seed"] = SEED

    df["model_config"] = MODEL_CONFIG
    df["sampling_json"] = json_col(SAMPLING)

    df["sbm_inits"] = SBM_INITS
    df["n_perm"] = N_PERM
    return df


def main(outdir: Path) -> None:
    write_frame(build_frame(), EXP, WILDCARD_COLUMNS, outdir)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("config/params"))
