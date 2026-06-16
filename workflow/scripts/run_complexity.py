"""
Thin wrapper around src.analysis.complexity_analysis.run_complexity_analysis.
Called by workflow/rules/analysis.smk rule run_complexity.
"""

import argparse
from pathlib import Path

from src.analysis.complexity_analysis import run_complexity_analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--posterior-ref",    type=Path, required=True,
                        help="posterior_samples.parquet from the reference model")
    parser.add_argument("--posterior-target", type=Path, required=True,
                        help="posterior_samples.parquet from the target model")
    parser.add_argument("--output-dir",       type=Path, required=True)
    parser.add_argument("--K",    type=int,   required=True)
    parser.add_argument("--kmax", type=int,   required=True)
    parser.add_argument("--reg",  type=float, default=1e-6)
    args = parser.parse_args()

    run_complexity_analysis(
        posterior_path_ref=args.posterior_ref,
        posterior_path_np=args.posterior_target,
        output_dir=args.output_dir,
        K=args.K,
        kmax=args.kmax,
        reg=args.reg,
    )


if __name__ == "__main__":
    main()
