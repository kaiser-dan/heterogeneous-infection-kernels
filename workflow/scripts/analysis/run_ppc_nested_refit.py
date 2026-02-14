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


def extract_mu_samples(draws: pd.DataFrame) -> np.ndarray:
    mu_cols = sorted(
        [c for c in draws.columns if c.startswith("mu[1,")],
        key=lambda c: int(c.split(",")[1].rstrip("]")),
    )
    if not mu_cols:
        raise ValueError("Could not find mu[1,*] columns in nonparam posterior draws.")
    return draws[mu_cols].to_numpy(dtype=float)


def stat_from_malahbois(mu_samples: np.ndarray, rho_samples: np.ndarray, stat_name: str) -> float:
    result = compute_malahbois_complexity(mu_samples=mu_samples, rho_samples=rho_samples)

    if stat_name == "mahal_non_centrality":
        return float(result.non_centrality)
    if stat_name == "mahal_median_total":
        return float(np.median(result.total_distance))
    if stat_name == "mahal_median_centered":
        return float(np.median(result.centered_distance))

    raise ValueError(
        "Unsupported nested statistic. Use one of: "
        "mahal_non_centrality,mahal_median_total,mahal_median_centered"
    )


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
    statistic: str,
):
    M_obs, N_obs = load_mn(summary_dir)
    Y = np.rint(M_obs + N_obs).astype(int)
    kmax = Y.shape[1]

    simple_obs_df = pd.read_parquet(simple_observed_posterior)
    nonparam_obs_df = pd.read_parquet(nonparam_observed_posterior)

    rho_obs = extract_rho_samples(simple_obs_df)
    mu_obs = extract_mu_samples(nonparam_obs_df)
    t_obs = stat_from_malahbois(mu_obs, rho_obs, statistic)

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
            pd.DataFrame(M_rep).to_parquet(rep_summary_dir / "M.parquet", index=False)
            pd.DataFrame(N_rep).to_parquet(rep_summary_dir / "N.parquet", index=False)

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
                t_rep = stat_from_malahbois(mu_rep, rho_rep, statistic)

                rows.append(
                    {
                        "rep_id": rep_id,
                        "draw_idx": int(draw_idx),
                        "rho_draw": rho_draw,
                        "stat_name": statistic,
                        "t_obs": float(t_obs),
                        "t_rep": float(t_rep),
                        "is_extreme": bool(t_rep >= t_obs),
                        "status": "ok",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    {
                        "rep_id": rep_id,
                        "draw_idx": int(draw_idx),
                        "rho_draw": rho_draw,
                        "stat_name": statistic,
                        "t_obs": float(t_obs),
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

    summary_df = pd.DataFrame(
        [
            {
                "stat_name": statistic,
                "n_reps_requested": int(draws),
                "n_reps_succeeded": int(len(valid)),
                "ppp_value": float(valid["is_extreme"].mean()),
                "t_obs": float(t_obs),
                "t_rep_median": float(valid["t_rep"].median()),
            }
        ]
    )
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
        "--statistic",
        type=str,
        default="mahal_non_centrality",
        help="One of: mahal_non_centrality,mahal_median_total,mahal_median_centered",
    )
    args = parser.parse_args()

    simple_params = json.loads(args.simple_instance_params)
    nonparam_params = json.loads(args.nonparam_instance_params)

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
        statistic=args.statistic,
    )


if __name__ == "__main__":
    main()
