from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp as _logsumexp

from src.inference.inference import fit_stan

@dataclass
class FoldResult:
    fold_idx: int
    cv_score: float
    n_held_out: int
    mean_log_pred: float

def compute_held_out_loglik(M_held: np.ndarray, N_held: np.ndarray, fit) -> dict:
    """Log posterior predictive density for held-out nodes.

    Uses population-level mu[k] and mixture weights pi, no per-node latents
    required. Correct marginal predictive for a new node from the population.
    """
    eps = 1e-9
    if M_held.shape[0] == 0:
        return {"cv_score": 0.0, "n_held_out": 0, "mean_log_pred": np.nan}

    mu = np.asarray(fit.stan_variable("mu"), dtype=float)   # (S, K, kmax) or (S, kmax)
    pi = np.asarray(fit.stan_variable("pi"), dtype=float)   # (S, K) or (S,)

    # Normalize to (S, K, kmax) and (S, K) regardless of K=1 shape from CmdStanPy.
    if mu.ndim == 2:
        mu = mu[:, np.newaxis, :]
    if pi.ndim == 1:
        pi = pi[:, np.newaxis]

    S, K, kmax_model = mu.shape
    assert pi.shape == (S, K)
    k = min(kmax_model, M_held.shape[1])

    mu_c    = np.clip(mu[:, :, :k], eps, 1.0 - eps)
    log_mu  = np.log(mu_c)
    log_1m  = np.log(1.0 - mu_c)
    log_pi  = np.log(np.clip(pi, eps, 1.0))

    M_b      = M_held[:, np.newaxis, np.newaxis, :k]
    N_b      = N_held[:, np.newaxis, np.newaxis, :k]
    log_mu_b = log_mu[np.newaxis, :, :, :]
    log_1m_b = log_1m[np.newaxis, :, :, :]

    ll_by_component = (M_b * log_mu_b + N_b * log_1m_b).sum(axis=-1)  # (V, S, K)
    lps      = ll_by_component + log_pi[np.newaxis, :, :]             # (V, S, K)
    log_lik  = _logsumexp(lps, axis=-1)                               # (V, S)
    log_pred = _logsumexp(log_lik, axis=-1) - np.log(S)               # (V,)

    cv_score = float(log_pred.sum())
    return {
        "cv_score":      cv_score,
        "n_held_out":    M_held.shape[0],
        "mean_log_pred": cv_score / M_held.shape[0],
    }


def _cv_fold_mask(V: int, cv_folds: int, cv_fold: int) -> np.ndarray:
    """Boolean mask of length V selecting held-out nodes for a given fold.

    Uses interleaved assignment (node i -> fold i % n_folds).
    """
    if cv_folds < 1:
        raise ValueError(f"cv_folds must be >= 1, got {cv_folds}")
    if not (0 <= cv_fold < cv_folds):
        raise ValueError(f"cv_fold={cv_fold} out of range [0, {cv_folds})")
    return (np.arange(V) % cv_folds) == cv_fold


def make_folds(V: int, n_folds: int) -> list[np.ndarray]:
    """Return list of boolean held-out masks, one per fold."""
    if n_folds < 1:
        raise ValueError(f"n_folds must be >= 1, got {n_folds}")
    return [_cv_fold_mask(V, n_folds, fold) for fold in range(n_folds)]


def fit_fold(
    held_mask: np.ndarray,
    fold_idx: int,
    M: np.ndarray,          # full matrix, held-out nodes included
    N: np.ndarray,          # full matrix, held-out nodes included
    config_path,
    instance_params=None,
) -> FoldResult:
    """Fit on train nodes, score on held-out nodes."""
    M_train, N_train = M[~held_mask], N[~held_mask]
    M_held,  N_held  = M[held_mask],  N[held_mask]

    print(f"CV fold {fold_idx}: train={M_train.shape[0]}, held={M_held.shape[0]}")

    stan_result = fit_stan(M_train, N_train, config_path, instance_params)
    result      = compute_held_out_loglik(M_held, N_held, stan_result.fit)

    return FoldResult(
        fold_idx=fold_idx,
        cv_score=result["cv_score"],
        n_held_out=result["n_held_out"],
        mean_log_pred=result["mean_log_pred"],
    )


def run_one_fold(
    M: np.ndarray,
    N: np.ndarray,
    n_folds: int,
    fold_idx: int,
    config_path,
    instance_params=None,
) -> FoldResult:
    """Fit + score a single CV fold (for parallelization)"""
    M = np.asarray(M, dtype=float)
    N = np.asarray(N, dtype=float)
    
    held_mask = _cv_fold_mask(M.shape[0], n_folds, fold_idx)
    
    return fit_fold(held_mask, fold_idx, M, N, config_path, instance_params)

def run_cv(
    M: np.ndarray,
    N: np.ndarray,
    n_folds: int,
    config_path,
    instance_params=None,
) -> list[FoldResult]:
    """Run full k-fold CV. Returns one FoldResult per fold."""
    M = np.asarray(M, dtype=float)
    N = np.asarray(N, dtype=float)

    folds   = make_folds(M.shape[0], n_folds)
    results = []
    for fold_idx, held_mask in enumerate(folds):
        results.append(fit_fold(held_mask, fold_idx, M, N, config_path, instance_params))
    return results
