"""Generator for config/params/cv_beta_sweep.tsv. Grid: beta2 x dgp x K x rep x fold.

CV model selection across kernel contrast r_beta = beta2 / beta1 (beta1 fixed at 0.01),
with fixed epsilon = 0.9, omega = 0.1, delta = 0.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv_sweep_frame import build_cv_sweep_frame
from gen_params_utils import axis_values, write_frame

EXP = "cv_beta_sweep"
AXES = ["beta2"]
WILDCARD_COLUMNS = AXES + ["dgp", "K", "rep", "fold"]

# ---- fixed ---------------------------------------------------------------------
BETA1 = 0.01
EPSILON = 0.9
OMEGA = 0.1
DELTA = 0.0

# ---- swept axis (points on beta_delta_sweep's beta2 axis, for comparability) ---
BETA2_AXIS = {"start": 0.02, "step": 0.01, "count": 4}  # 0.02, 0.03, 0.04, 0.05


def build_frame():
    cells = [
        {"epsilon": EPSILON, "omega": OMEGA, "beta1": BETA1, "beta2": b, "delta": DELTA}
        for b in axis_values(BETA2_AXIS)
    ]
    return build_cv_sweep_frame(cells, AXES)


def main(outdir: Path) -> None:
    write_frame(build_frame(), EXP, WILDCARD_COLUMNS, outdir)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("config/params"))
