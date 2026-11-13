"""Generator for config/params/cv_epsilon_omega_sweep.tsv. Grid: epsilon x omega x dgp x K x rep x fold.

CV model selection across community separation (epsilon), kernel-community alignment (omega)
with fixed dynamics (beta1 = 0.01, beta2 = 0.04, delta = 0).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv_sweep_frame import build_cv_sweep_frame
from gen_params_utils import axis_values, write_frame

EXP = "cv_epsilon_omega_sweep"
AXES = ["epsilon", "omega"]
WILDCARD_COLUMNS = AXES + ["dgp", "K", "rep", "fold"]

# ---- fixed ---------------------------------------------------------------------
BETA1 = 0.01
BETA2 = 0.04
DELTA = 0.0

# ---- swept axes (points on epsilon_omega_sweep's grid, for comparability) -------
EPSILON_AXIS = {"start": 0.1, "step": 0.2, "count": 5}  # 0.1, 0.3, 0.5, 0.7, 0.9
OMEGA_AXIS = {"start": 0.0, "step": 0.1, "count": 6}    # 0.0, 0.1, 0.2, 0.3, 0.4, 0.5


def build_frame():
    cells = [
        {"epsilon": e, "omega": w, "beta1": BETA1, "beta2": BETA2, "delta": DELTA}
        for e in axis_values(EPSILON_AXIS)
        for w in axis_values(OMEGA_AXIS)
    ]
    return build_cv_sweep_frame(cells, AXES)


def main(outdir: Path) -> None:
    write_frame(build_frame(), EXP, WILDCARD_COLUMNS, outdir)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("config/params"))
