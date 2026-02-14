import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def simple_kernel(rho: float, kmax: int) -> np.ndarray:
    nu = np.arange(1, kmax + 1, dtype=float)
    return 1.0 - (1.0 - float(rho)) ** nu


def load_mn(summary_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    if not summary_dir.exists() or not summary_dir.is_dir():
        raise ValueError(f"Expected summary directory with M/N parquet files: {summary_dir}")

    m_path = summary_dir / "M.parquet"
    n_path = summary_dir / "N.parquet"
    if not m_path.exists() or not n_path.exists():
        raise ValueError(f"Missing M.parquet or N.parquet in summary directory: {summary_dir}")

    M = pd.read_parquet(m_path).values
    N = pd.read_parquet(n_path).values
    if M.shape != N.shape:
        raise ValueError("M and N must have identical shape.")
    return M.astype(float), N.astype(float)


def extract_rho_samples(draws: pd.DataFrame) -> np.ndarray:
    if "rho[1]" in draws.columns:
        return draws["rho[1]"].to_numpy(dtype=float)

    rho_cols = [c for c in draws.columns if c.startswith("rho[")]
    if len(rho_cols) == 1:
        return draws[rho_cols[0]].to_numpy(dtype=float)

    if "rho" in draws.columns:
        return draws["rho"].to_numpy(dtype=float)

    raise ValueError("Could not find rho samples in simple posterior draws.")


def stat_deviance(M: np.ndarray, N: np.ndarray, p: np.ndarray, eps: float = 1e-9) -> float:
    Y = M + N
    p = np.clip(p, eps, 1.0 - eps)
    denom = (Y * p * (1.0 - p)) + eps
    return float(np.sum((M - (Y * p)) ** 2 / denom))


def stat_neg_loglik(M: np.ndarray, N: np.ndarray, p: np.ndarray, eps: float = 1e-9) -> float:
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.sum(M * np.log(p) + N * np.log(1.0 - p)))


def stat_l1_rate(M: np.ndarray, N: np.ndarray, p: np.ndarray, eps: float = 1e-9) -> float:
    Y = M + N
    empirical = np.where(Y > 0, M / (Y + eps), 0.0)
    return float(np.sum(np.abs(empirical - p[None, :]) * (Y > 0)))


STAT_REGISTRY = {
    "deviance": stat_deviance,
    "neg_loglik": stat_neg_loglik,
    "l1_rate": stat_l1_rate,
}


# All currently-registered statistics are "larger = worse fit".
HIGHER_IS_MORE_EXTREME = {name: True for name in STAT_REGISTRY.keys()}


def parse_stats(stats_arg: str) -> list[str]:
    stats = [s.strip() for s in stats_arg.split(",") if s.strip()]
    if not stats:
        raise ValueError("No PPC statistics were provided.")

    invalid = [s for s in stats if s not in STAT_REGISTRY]
    if invalid:
        raise ValueError(
            f"Unknown statistic(s): {invalid}. Valid options: {sorted(STAT_REGISTRY.keys())}"
        )
    return stats


def run_non_nested_ppc(
    summary_dir: Path,
    simple_posterior: Path,
    details_output: Path,
    summary_output: Path,
    draws: int,
    seed: int,
    stats: list[str],
):
    M_obs, N_obs = load_mn(summary_dir)
    kmax = M_obs.shape[1]

    draws_df = pd.read_parquet(simple_posterior)
    rho = extract_rho_samples(draws_df)
    if rho.size == 0:
        raise ValueError("Simple posterior has zero rho samples.")

    rng = np.random.default_rng(seed)
    replace = rho.size < draws
    draw_indices = rng.choice(rho.size, size=draws, replace=replace)

    Y = np.rint(M_obs + N_obs).astype(int)

    rows = []
    for rep_id, draw_idx in enumerate(draw_indices):
        rho_draw = float(rho[draw_idx])
        p = np.clip(simple_kernel(rho_draw, kmax), 1e-9, 1.0 - 1e-9)

        M_rep = rng.binomial(Y, p[None, :]).astype(float)
        N_rep = (Y - M_rep).astype(float)

        for stat_name in stats:
            stat_fn = STAT_REGISTRY[stat_name]
            t_obs = float(stat_fn(M_obs, N_obs, p))
            t_rep = float(stat_fn(M_rep, N_rep, p))
            is_extreme = t_rep >= t_obs if HIGHER_IS_MORE_EXTREME[stat_name] else t_rep <= t_obs

            rows.append(
                {
                    "rep_id": rep_id,
                    "draw_idx": int(draw_idx),
                    "rho_draw": rho_draw,
                    "stat_name": stat_name,
                    "t_obs": t_obs,
                    "t_rep": t_rep,
                    "is_extreme": bool(is_extreme),
                }
            )

    details_df = pd.DataFrame(rows)
    details_output.parent.mkdir(parents=True, exist_ok=True)
    details_df.to_parquet(details_output, index=False)

    summary_df = (
        details_df.groupby("stat_name", as_index=False)
        .agg(
            n_reps=("rep_id", "count"),
            ppp_value=("is_extreme", "mean"),
            t_obs_median=("t_obs", "median"),
            t_rep_median=("t_rep", "median"),
        )
        .sort_values("stat_name")
    )
    summary_df.to_parquet(summary_output, index=False)


def main():
    parser = argparse.ArgumentParser(description="Run non-nested PPC from simple-model posterior draws.")
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--simple-posterior", type=Path, required=True)
    parser.add_argument("--details-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--stats",
        type=str,
        default="deviance,neg_loglik,l1_rate",
        help="Comma-separated list from: deviance,neg_loglik,l1_rate",
    )
    args = parser.parse_args()

    stats = parse_stats(args.stats)
    run_non_nested_ppc(
        summary_dir=args.summary_dir,
        simple_posterior=args.simple_posterior,
        details_output=args.details_output,
        summary_output=args.summary_output,
        draws=int(args.draws),
        seed=int(args.seed),
        stats=stats,
    )


if __name__ == "__main__":
    main()
