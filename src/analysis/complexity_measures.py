from __future__ import annotations

import numpy as np
from scipy.special import logit

def effective_rank(covariance: np.ndarray, reg: float = 1e-6) -> float:
    """Effective rank of the *precision* of a covariance matrix.

    Parameters
    ----------
    covariance : (k, k) array
        Positive semi-definite covariance matrix.
    reg : float
        Relative ridge: adds reg * trace(covariance) / k to each diagonal
        entry before computing eigenvalues. Scale-invariant.

    Reference: Roy & Vetterli, EUSIPCO 2007.
    """
    sigma = np.asarray(covariance, dtype=float)
    k = sigma.shape[0]
    sigma = sigma + (reg * np.trace(sigma) / k) * np.eye(k)
    return _erank_from_regularized(sigma)


def _erank_from_regularized(sigma: np.ndarray) -> float:
    """Effective rank of the precision from an already-regularized covariance."""
    eigs = np.linalg.eigvalsh(sigma)
    log_prec = -np.log(eigs)
    log_prec -= log_prec.max()
    p = np.exp(log_prec)
    p /= p.sum()
    return float(np.exp(-np.sum(p * np.log(p + 1e-300))))


def mahalanobis_complexity_logit(
    mu_post: np.ndarray,
    ref: np.ndarray,
    reg: float = 1e-6,
    eps: float = 1e-8,
) -> tuple[np.ndarray, float, float]:
    """Mahalanobis-distance complexity on the logit scale.

    Parameters
    ----------
    mu_post : (n_samples, kmax) array
        Posterior draws of the infection kernel on the *probability* scale.
        Clipped to (eps, 1-eps) before logit transform.
    ref : (kmax,) array
        Reference kernel on the *logit* scale. 
    reg : float
        Relative ridge regularization applied to the posterior covariance.
    eps : float
        Clip boundary for the internal logit transform.

    Returns
    -------
    distances : (n_samples,) array
        Squared Mahalanobis distance of each draw from ref.
    mean_distance : float
        Mean squared Mahalanobis distance across posterior draws.
    erank : float
        Effective rank of the posterior precision on the logit scale.
        Divide mean_distance by erank for the normalized complexity score.
    """
    mu_post = np.asarray(mu_post, dtype=float)
    ref = np.asarray(ref, dtype=float).ravel()
    if mu_post.ndim != 2:
        raise ValueError("mu_post must be 2D (n_samples, kmax)")
    if mu_post.shape[1] != ref.shape[0]:
        raise ValueError(
            f"mu_post has kmax={mu_post.shape[1]} but ref has length {ref.shape[0]}"
        )
    if mu_post.shape[0] < 2:
        raise ValueError("Need at least 2 posterior samples")

    X = logit(np.clip(mu_post, eps, 1.0 - eps))

    S, K = X.shape
    X_c = X - X.mean(axis=0)
    sigma = (X_c.T @ X_c) / (S - 1)
    sigma += (reg * np.trace(sigma) / K) * np.eye(K)

    L = np.linalg.cholesky(sigma)
    diff = X - ref[None, :]
    z = np.linalg.solve(L, diff.T)
    distances = np.sum(z ** 2, axis=0)

    erank = _erank_from_regularized(sigma)
    return distances, float(distances.mean()), erank