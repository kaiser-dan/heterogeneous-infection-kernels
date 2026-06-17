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