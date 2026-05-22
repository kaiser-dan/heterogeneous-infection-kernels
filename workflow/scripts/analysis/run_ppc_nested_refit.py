import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.complexity_measures import compute_malahbois_complexity  # noqa: E402
from src.models.inference import run_stan_inference  # noqa: E402


SUPPORTED_STAT_NAMES = (
    "mahal_non_centrality",
    "mahal_mode_non_centrality",
    "mahal_median_total",
    "mahal_median_centered",
)


def simple_kernel(rho: float, kmax: int) -> np.ndarray:
    nu = np.arange(1, kmax + 1, dtype=float)
    return 1.0 - (1.0 - float(rho)) ** nu


def _sorted_value_cols(columns) -> list[str]:
    value_cols = [str(column) for column in columns if str(column) != "user_id"]
    return sorted(value_cols, key=lambda col: int(col) if str(col).isdigit() else str(col))


def _read_matrix_parquet(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    df = pd.read_parquet(path).copy()
    if "user_id" in df.columns:
        user_ids = df["user_id"].astype(str).tolist()
    else:
        user_ids = [str(idx) for idx in range(len(df))]
        df.insert(0, "user_id", user_ids)

    value_cols = _sorted_value_cols(df.columns)
    values = (
        df[value_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    return user_ids, value_cols, values


def load_mn(summary_dir: Path) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    if not summary_dir.exists() or not summary_dir.is_dir():
        raise ValueError(f"Expected summary directory with M/N parquet files: {summary_dir}")

    m_path = summary_dir / "M.parquet"
    n_path = summary_dir / "N.parquet"
    if not m_path.exists() or not n_path.exists():
        raise ValueError(f"Missing M.parquet or N.parquet in summary directory: {summary_dir}")

    user_ids, value_cols, M = _read_matrix_parquet(m_path)
    n_user_ids, n_value_cols, N = _read_matrix_parquet(n_path)
    if user_ids != n_user_ids or value_cols != n_value_cols:
        raise ValueError("M and N must share identical user_id ordering and exposure columns.")
    if M.shape != N.shape:
        raise ValueError("M and N must have identical shape.")
    return user_ids, value_cols, M.astype(float), N.astype(float)


def extract_rho_samples(draws: pd.DataFrame) -> np.ndarray:
    if "rho[1]" in draws.columns:
        return draws["rho[1]"].to_numpy(dtype=float)

    rho_cols = [c for c in draws.columns if c.startswith("rho[")]
    if len(rho_cols) == 1:
        return draws[rho_cols[0]].to_numpy(dtype=float)

    if "rho" in draws.columns:
        return draws["rho"].to_numpy(dtype=float)

    raise ValueError("Could not find rho samples in simple posterior draws.")


def extract_mu_samples(draws: pd.DataFrame) -> np.ndarray:
    mu_cols = sorted(
        [c for c in draws.columns if c.startswith("mu[1,")],
        key=lambda c: int(c.split(",")[1].rstrip("]")),
    )
    if not mu_cols:
        raise ValueError("Could not find mu[1,*] columns in nonparam posterior draws.")
    return draws[mu_cols].to_numpy(dtype=float)


def stat_values_from_malahbois(mu_samples: np.ndarray, rho_samples: np.ndarray) -> dict[str, float]:
    result = compute_malahbois_complexity(mu_samples=mu_samples, rho_samples=rho_samples)
    return {
        "mahal_non_centrality": float(result.non_centrality),
        "mahal_mode_non_centrality": float(result.mode_non_centrality),
        "mahal_median_total": float(np.median(result.total_distance)),
        "mahal_median_centered": float(np.median(result.centered_distance)),
    }


def stat_from_malahbois(mu_samples: np.ndarray, rho_samples: np.ndarray, stat_name: str) -> float:
    return stat_values_from_malahbois(mu_samples=mu_samples, rho_samples=rho_samples)[stat_name]


def parse_stats(stats_arg: str) -> list[str]:
    stats = [s.strip() for s in stats_arg.split(",") if s.strip()]
    if not stats:
        raise ValueError("No nested PPC statistics were provided.")

    invalid = [s for s in stats if s not in SUPPORTED_STAT_NAMES]
    if invalid:
        raise ValueError(
            f"Unknown statistic(s): {invalid}. Valid options: {list(SUPPORTED_STAT_NAMES)}"
        )
    return stats


def fit_one_dataset(
    summary_dir: Path,
    output_dir: Path,
    config_path: Path,
    instance_params: dict,
) -> pd.DataFrame:
    run_stan_inference(
        summary_path=str(summary_dir),
        output_dir=str(output_dir),
        instance_params=instance_params,
        config_path=str(config_path),
    )
    posterior_path = output_dir / "posterior_samples.parquet"
    if not posterior_path.exists():
        raise RuntimeError(f"Missing posterior samples at {posterior_path}")
    return pd.read_parquet(posterior_path)


def run_nested_ppc(
    summary_dir: Path,
    config_path: Path,
    simple_observed_posterior: Path,
    nonparam_observed_posterior: Path,
    simple_instance_params: dict,
    nonparam_instance_params: dict,
    details_output: Path,
    summary_output: Path,
    draws: int,
    seed: int,
    stats: list[str],
):
    user_ids, value_cols, M_obs, N_obs = load_mn(summary_dir)
    Y = np.rint(M_obs + N_obs).astype(int)
    kmax = Y.shape[1]

    simple_obs_df = pd.read_parquet(simple_observed_posterior)
    nonparam_obs_df = pd.read_parquet(nonparam_observed_posterior)

    rho_obs = extract_rho_samples(simple_obs_df)
    mu_obs = extract_mu_samples(nonparam_obs_df)
    t_obs_by_stat = stat_values_from_malahbois(mu_obs, rho_obs)

    rng = np.random.default_rng(seed)
    replace = rho_obs.size < draws
    draw_indices = rng.choice(rho_obs.size, size=draws, replace=replace)

    rows = []
    with tempfile.TemporaryDirectory(prefix="ppc_nested_") as tmp_root_str:
        tmp_root = Path(tmp_root_str)
        for rep_id, draw_idx in enumerate(draw_indices):
            rho_draw = float(rho_obs[draw_idx])
            p = np.clip(simple_kernel(rho_draw, kmax), 1e-9, 1.0 - 1e-9)
            M_rep = rng.binomial(Y, p[None, :]).astype(float)
            N_rep = (Y - M_rep).astype(float)

            rep_summary_dir = tmp_root / f"rep_{rep_id}" / "summary"
            rep_summary_dir.mkdir(parents=True, exist_ok=True)
            m_rep_df = pd.DataFrame(M_rep, columns=value_cols)
            n_rep_df = pd.DataFrame(N_rep, columns=value_cols)
            m_rep_df.insert(0, "user_id", user_ids)
            n_rep_df.insert(0, "user_id", user_ids)
            m_rep_df.to_parquet(rep_summary_dir / "M.parquet", index=False)
            n_rep_df.to_parquet(rep_summary_dir / "N.parquet", index=False)

            rep_simple_dir = tmp_root / f"rep_{rep_id}" / "fit_simple"
            rep_nonparam_dir = tmp_root / f"rep_{rep_id}" / "fit_nonparam"

            try:
                simple_rep_df = fit_one_dataset(
                    summary_dir=rep_summary_dir,
                    output_dir=rep_simple_dir,
                    config_path=config_path,
                    instance_params=simple_instance_params,
                )
                nonparam_rep_df = fit_one_dataset(
                    summary_dir=rep_summary_dir,
                    output_dir=rep_nonparam_dir,
                    config_path=config_path,
                    instance_params=nonparam_instance_params,
                )

                rho_rep = extract_rho_samples(simple_rep_df)
                mu_rep = extract_mu_samples(nonparam_rep_df)
                t_rep_by_stat = stat_values_from_malahbois(mu_rep, rho_rep)

                for stat_name in stats:
                    t_obs = float(t_obs_by_stat[stat_name])
                    t_rep = float(t_rep_by_stat[stat_name])
                    rows.append(
                        {
                            "rep_id": rep_id,
                            "draw_idx": int(draw_idx),
                            "rho_draw": rho_draw,
                            "stat_name": stat_name,
                            "t_obs": t_obs,
                            "t_rep": t_rep,
                            "is_extreme": bool(t_rep >= t_obs),
                            "status": "ok",
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                for stat_name in stats:
                    rows.append(
                        {
                            "rep_id": rep_id,
                            "draw_idx": int(draw_idx),
                            "rho_draw": rho_draw,
                            "stat_name": stat_name,
                            "t_obs": float(t_obs_by_stat[stat_name]),
                            "t_rep": np.nan,
                            "is_extreme": False,
                            "status": f"error:{type(exc).__name__}",
                        }
                    )

    details_df = pd.DataFrame(rows)
    details_output.parent.mkdir(parents=True, exist_ok=True)
    details_df.to_parquet(details_output, index=False)

    valid = details_df[details_df["status"] == "ok"].copy()
    if valid.empty:
        raise RuntimeError("All nested PPC replicates failed during re-fit.")

    summary_df = (
        valid.groupby("stat_name", as_index=False)
        .agg(
            n_reps_succeeded=("rep_id", "count"),
            ppp_value=("is_extreme", "mean"),
            t_obs=("t_obs", "median"),
            t_rep_median=("t_rep", "median"),
        )
        .sort_values("stat_name")
    )
    summary_df.insert(1, "n_reps_requested", int(draws))
    summary_df.to_parquet(summary_output, index=False)


def main():
    parser = argparse.ArgumentParser(description="Run nested re-fit PPC for pairwise-fit statistics.")
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--simple-observed-posterior", type=Path, required=True)
    parser.add_argument("--nonparam-observed-posterior", type=Path, required=True)
    parser.add_argument("--simple-instance-params", type=str, required=True)
    parser.add_argument("--nonparam-instance-params", type=str, required=True)
    parser.add_argument("--details-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--stats",
        type=str,
        default=None,
        help=(
            "Comma-separated list from: "
            "mahal_non_centrality,mahal_mode_non_centrality,mahal_median_total,mahal_median_centered"
        ),
    )
    parser.add_argument(
        "--statistic",
        type=str,
        default=None,
        help="Backward-compatible singular alias for --stats.",
    )
    args = parser.parse_args()

    simple_params = json.loads(args.simple_instance_params)
    nonparam_params = json.loads(args.nonparam_instance_params)
    stats_arg = args.stats if args.stats is not None else (args.statistic or "mahal_non_centrality")
    stats = parse_stats(stats_arg)

    run_nested_ppc(
        summary_dir=args.summary_dir,
        config_path=args.config,
        simple_observed_posterior=args.simple_observed_posterior,
        nonparam_observed_posterior=args.nonparam_observed_posterior,
        simple_instance_params=simple_params,
        nonparam_instance_params=nonparam_params,
        details_output=args.details_output,
        summary_output=args.summary_output,
        draws=int(args.draws),
        seed=int(args.seed),
        stats=stats,
    )


if __name__ == "__main__":
    main()
