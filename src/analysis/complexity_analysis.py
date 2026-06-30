from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit

from .complexity_measures import (
    mahalanobis_complexity_logit,
)


def posterior_mean_logit(mu_draws: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Posterior mean on the logit scale from probability-scale draws.
 
    Parameters
    ----------
    mu_draws : (n_samples, kmax) array on the probability scale.
    eps : float
        Clip boundary before logit transform.
 
    Returns
    -------
    (kmax,) array on the logit scale.
    """
    return logit(np.clip(mu_draws, eps, 1.0 - eps)).mean(axis=0)


def compute_complexity_scores(
    mu_np_draws: np.ndarray,
    ref: np.ndarray,
    reg: float = 1e-6,
) -> dict:
    """Compute Mahalanobis complexity score for a single-component kernel posterior.

    Parameters
    ----------
    mu_np_draws : (n_draws, kmax) or (n_draws, 1, kmax) array on the *probability* scale
        Posterior draws from a single mixture component.
    ref : (kmax,) array on the *logit* scale
        E.g., reference kernel from posterior_mean_logit.
    reg : float
        Ridge regularization passed to mahalanobis_complexity_logit.

    Returns
    -------
    dict with keys: mean_distance, erank, complexity, n_draws, kmax
    """
    mu_np_draws = np.asarray(mu_np_draws, dtype=float)
    if mu_np_draws.ndim == 3:
        if mu_np_draws.shape[1] != 1:
            raise ValueError(
                f"mu_np_draws has {mu_np_draws.shape[1]} components; "
                "compute_complexity_scores expects a single component. "
                "Slice with mu_np_draws[:, k, :] before calling."
            )
        mu_np_draws = mu_np_draws[:, 0, :]

    if mu_np_draws.ndim != 2:
        raise ValueError(f"mu_np_draws must be 2D (n_draws, kmax), got shape {mu_np_draws.shape}")

    n_draws, kmax = mu_np_draws.shape
    ref = np.asarray(ref, dtype=float).squeeze()
    if ref.ndim != 1:
        raise ValueError(f"ref must be a 1D vector, got shape {ref.shape}")
    if ref.shape[0] != kmax:
        raise ValueError(f"ref length {ref.shape[0]} does not match kmax={kmax}")

    distances, mean_dist, erank = mahalanobis_complexity_logit(mu_np_draws, ref, reg=reg)

    return {
        "mean_distance": float(mean_dist),
        "erank":         float(erank),
        "complexity":    float(mean_dist / erank),
        "n_draws":       int(n_draws),
        "kmax":          int(kmax),
    }


def simple_reference_logit(beta: float, kmax: int, eps: float = 1e-8) -> np.ndarray:
    """Logit-scale simple contagion reference kernel.

    mu_sc(nu) = 1 - (1 - beta)^nu, nu = 1..kmax, returned on the logit scale.
    This is the best-fitting simple kernel reference once beta = beta_hat.
    """
    nu = np.arange(1, int(kmax) + 1, dtype=float)
    curve = 1.0 - (1.0 - float(beta)) ** nu
    return logit(np.clip(curve, eps, 1.0 - eps))


def ppc_pvalue(d_obs: float, d_null, plus_one: bool = False) -> dict:
    """Posterior-predictive p-value: fraction of null complexities >= observed.

        plus_one=False : p = #{D_null >= D_obs} / R
        plus_one=True  : p = (1 + #{D_null >= D_obs}) / (R+1)  (avoids p=0; the
                         standard Monte-Carlo estimator, recommended for finite R)

    NaN replicates (failed fits) are dropped; n_null reports the effective R.
    """
    d_null = np.asarray(d_null, dtype=float).ravel()
    d_null = d_null[np.isfinite(d_null)]
    R = int(d_null.size)
    if R == 0:
        return {"p_value": float("nan"), "n_null": 0, "n_ge": 0, "d_obs": float(d_obs)}
    
    n_ge = int(np.count_nonzero(d_null >= d_obs))
    p = (1 + n_ge) / (R + 1) if plus_one else n_ge / R
    return {"p_value": float(p), "n_null": R, "n_ge": n_ge, "d_obs": float(d_obs)}


def ppc_complexity(
    obs_mu_draws: np.ndarray,
    null_mu_draws,
    beta: float,
    reg: float = 1e-6,
    eps: float = 1e-8,
    plus_one: bool = False,
) -> dict:
    """Full complexity PPC from in-memory posterior draws.

    Scores the observed np draws and each null replicate's np draws against the
    SAME geometric reference (built from beta = beta_hat used to generate the
    nulls), then returns the p-value. Does NOT simulate or fit -- you supply the
    null draws.

    Parameters
    ----------
    obs_mu_draws  : (S, kmax) or (S, 1, kmax) np posterior draws, observed data.
    null_mu_draws : iterable of (S_r, kmax) arrays, one per null replicate.
    beta          : beta_hat -- generates the nulls and defines the reference.

    Returns dict: p_value, n_null, n_ge, d_obs, d_null (array of replicate scores).
    """
    obs = np.asarray(obs_mu_draws, dtype=float)
    if obs.ndim == 3:
        obs = obs[:, 0, :]
    kmax = obs.shape[1]
    ref = simple_reference_logit(beta, kmax, eps=eps)

    d_obs = compute_complexity_scores(obs, ref, reg=reg)["complexity"]

    d_null = []
    for nd in null_mu_draws:
        nd = np.asarray(nd, dtype=float)
        if nd.ndim == 3:
            nd = nd[:, 0, :]
        try:
            d_null.append(compute_complexity_scores(nd, ref, reg=reg)["complexity"])
        except Exception:
            d_null.append(np.nan)

    out = ppc_pvalue(d_obs, d_null, plus_one=plus_one)
    out["d_null"] = np.asarray(d_null, dtype=float)
    return out
