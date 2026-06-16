from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .complexity_measures import (
    mahalanobis_complexity_logit,
    posterior_mean_logit,
)


def load_mu_draws(posterior_path: str | Path, K: int, kmax: int) -> np.ndarray:
    """Load component kernel draws from posterior_samples.parquet.

    Parameters
    ----------
    posterior_path : path to posterior_samples.parquet
    K : int
        Number of mixture components.
    kmax : int
        Maximum exposure level.

    Returns
    -------
    (n_draws, K, kmax) float array on the probability scale.
    """
    df = pd.read_parquet(posterior_path)

    mu_cols = sorted(
        [c for c in df.columns if re.match(r"^mu\[\d+,\d+\]$", c)],
        key=lambda c: tuple(int(x) for x in re.findall(r"\d+", c)),
    )
    if not mu_cols:
        raise ValueError(
            f"No mu[k,nu] columns found in {posterior_path}. "
            "Was the model run with posterior_mode='full'?"
        )
    if len(mu_cols) != K * kmax:
        raise ValueError(
            f"Expected {K * kmax} mu columns (K={K}, kmax={kmax}), "
            f"found {len(mu_cols)}."
        )

    return df[mu_cols].values.reshape(-1, K, kmax).astype(float)


def logit_reference_from_posterior(
    mu_ref_draws: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Build the (logit-scale) reference kernel from the reference model's posterior.

    Parameters
    ----------
    mu_ref_draws : (n_draws, K, kmax) array on the probability scale.
    eps : float
        Clip boundary before logit transform.

    Returns
    -------
    (kmax,) reference kernel on the logit scale.
    """
    n_draws, K, kmax = mu_ref_draws.shape

    if K > 1:
        warnings.warn(
            f"reference_from_posterior: K={K} components are being pooled into a "
            "single reference kernel by averaging on the logit scale.",
            UserWarning,
            stacklevel=2,
        )

    # Pool over draws and components: (n_draws, K, kmax) -> (n_draws * K, kmax)
    pooled = mu_ref_draws.reshape(-1, kmax)
    return posterior_mean_logit(pooled, eps=eps)


def compute_complexity_scores(
    mu_np_draws: np.ndarray,
    ref: np.ndarray,
    reg: float = 1e-6,
) -> pd.DataFrame:
    """Compute Mahalanobis complexity scores for each non-parametric component.

    Parameters
    ----------
    mu_np_draws : (n_draws, K, kmax) array on the *probability* scale
        Posterior draws from non_parametric_monotonic_empirical_bayes.
    ref : (kmax,) array on the *logit* scale
        E.g., reference kernel from logit_reference_from_posterior.
    reg : float
        Ridge regularization passed to mahalanobis_complexity_logit.

    Returns
    -------
    DataFrame with one row per component and columns:
        component, mean_distance, erank, complexity, n_draws, kmax
    """
    n_draws, K, kmax = mu_np_draws.shape
    rows = []
    for k in range(K):
        distances, mean_dist, erank = mahalanobis_complexity_logit(
            mu_np_draws[:, k, :],
            ref,
            reg=reg,
        )
        rows.append({
            "component":     int(k + 1),
            "mean_distance": float(mean_dist),
            "erank":         float(erank),
            "complexity":    float(mean_dist / erank),
            "n_draws":       int(n_draws),
            "kmax":          int(kmax),
        })

    return pd.DataFrame(rows)


def run_complexity_analysis(
    posterior_path_ref: str | Path,
    posterior_path_np: str | Path,
    output_dir: str | Path,
    K: int,
    kmax: int,
    reg: float = 1e-6,
    parquet_compression: str = "zstd",
) -> pd.DataFrame:
    """Load inference output from both models, compute complexity, write results.

    Parameters
    ----------
    posterior_path_ref : path to posterior_samples.parquet from reference model
    posterior_path_np : path to posterior_samples.parquet from non_parametric_monotonic_eb
    output_dir : directory to write complexity_scores.parquet
    K : int
        Number of mixture components (must be the same for both models).
    kmax : int
        Maximum exposure level.
    reg : float
        Ridge regularization for covariance estimation.
    parquet_compression : str
        Parquet compression codec.

    Returns
    -------
    DataFrame of per-component complexity scores (also written to output_dir).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mu_ref_draws = load_mu_draws(posterior_path_ref, K=K, kmax=kmax)
    mu_np_draws  = load_mu_draws(posterior_path_np,  K=K, kmax=kmax)

    ref    = logit_reference_from_posterior(mu_ref_draws)
    scores = compute_complexity_scores(mu_np_draws, ref, reg=reg)

    out_path = output_dir / "complexity_scores.parquet"
    scores.to_parquet(out_path, index=False, compression=parquet_compression)
    print(f"Complexity scores written to {out_path}")
    for _, row in scores.iterrows():
        print(
            f"  Component {int(row.component)}: "
            f"complexity={row.complexity:.4f}  "
            f"erank={row.erank:.2f}  "
            f"mean_distance={row.mean_distance:.4f}"
        )

    return scores
