"""
Compile Stan models in src/models/ using CmdStanPy.

Compiled binaries are written alongside their .stan files (CmdStanPy default).
Run from the repo root via `just compile` or directly:

    uv run python bin/compile_models.py                        # compile all
    uv run python bin/compile_models.py mixture_simple.stan  # one model
"""

import sys
from pathlib import Path

import cmdstanpy

MODELS_DIR = Path(__file__).resolve().parent.parent / "src" / "models"


def compile_model(stan_path: Path) -> None:
    print(f"Compiling {stan_path.name} ...")
    cmdstanpy.CmdStanModel(stan_file=stan_path, force_compile=True)
    print(f"  OK: {stan_path.with_suffix('')}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Compile a single named model
        stan_path = MODELS_DIR / sys.argv[1]
        if not stan_path.suffix:
            stan_path = stan_path.with_suffix(".stan")
        if not stan_path.exists():
            raise FileNotFoundError(f"Stan file not found: {stan_path}")
        compile_model(stan_path)
    else:
        # Compile all .stan files in src/models/
        stan_files = sorted(MODELS_DIR.glob("*.stan"))
        if not stan_files:
            raise FileNotFoundError(f"No .stan files found in {MODELS_DIR}")
        for stan_path in stan_files:
            compile_model(stan_path)
        print(f"\nCompiled {len(stan_files)} model(s).")
