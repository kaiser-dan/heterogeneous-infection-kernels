"""Regenerate every experiment's committed Paramspace frame into a target directory."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gen_params_beta_delta_sweep as gen_params_beta_delta_sweep
import gen_params_congress_omega_sweep as gen_params_congress_omega_sweep
import gen_params_cv_basic as gen_params_cv_basic
import gen_params_epsilon_omega_sweep as gen_params_epsilon_omega_sweep
import gen_params_ppc as gen_params_ppc

GENERATORS = [
    gen_params_beta_delta_sweep,
    gen_params_epsilon_omega_sweep,
    gen_params_congress_omega_sweep,
    gen_params_cv_basic,
    gen_params_ppc,
]


def main(outdir: Path) -> None:
    for gen in GENERATORS:
        gen.main(outdir)
        print(f"wrote {outdir / gen.EXP}.tsv")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("config/params"))
