from __future__ import annotations

import numpy as np
from scipy.special import logit
from scipy.linalg import solve_triangular
from sklearn.covariance import LedoitWolf


def complexity_score_components(
    mu_post: np.ndarray,
    ref: np.ndarray,
    reg: float = 1e-2,
    eps: float = 1e-8
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
    trace_p_sigma: float
        Posterior-spread portion of mean_distance.
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
    Sigma = (X_c.T @ X_c) / (S-1) # sample covariance, no ridge
    Sigma_reg = Sigma + (reg * np.trace(Sigma) / K) * np.eye(K)

    L = np.linalg.cholesky(Sigma_reg)
    diff = X - ref[None, :]
    z = solve_triangular(L, diff.T, lower=True)
    distances = np.sum(z ** 2, axis=0) # (X - r)^T P (X - r) per draw

    # tr(P Sigma) = tr(inv(sigma_reg) @ sigma), reusing the Cholesky factor.
    Zc = solve_triangular(L, X_c.T, lower=True)
    trace_p_sigma = float(np.sum(Zc**2) / (S - 1))
    
    return distances, float(distances.mean()), trace_p_sigma


def _erank_from_regularized(sigma: np.ndarray) -> float:
    """Effective rank of the precision from an already-regularized covariance."""
    eigs = np.linalg.eigvalsh(sigma)
    log_prec = -np.log(eigs)
    log_prec -= log_prec.max()
    p = np.exp(log_prec)
    p /= p.sum()
    return float(np.exp(-np.sum(p * np.log(p + 1e-300))))


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


def effective_ledoit_wolf_shrinkage(
    mu_draws: np.ndarray,
    eps: float = 1e-8,
) -> dict:
    """Effective ridge that Ledoit-Wolf shrinkage corresponds to.

    Parameters
    ----------
    mu_draws : (n_draws, kmax) or (n_draws, 1, kmax) array
        Posterior draws on the *probability* scale (a single mixture component).
        Clipped to (eps, 1-eps) and logit-transformed before fitting LW, matching
        the covariance convention used elsewhere in this module.
    eps : float
        Clip boundary for the internal logit transform.

    Returns
    -------
    dict with keys:
        shrinkage : float
            LW shrinkage intensity delta in [0, 1] (sklearn's LedoitWolf.shrinkage_).
        reg_eff : float
            Effective relative ridge delta / (1 - delta). 
            Larger data-driven shrinkage -> larger effective ridge.
        n_draws : int
            Number of posterior draws fitted.
        kmax : int
            Kernel dimension.
    """
    mu_draws = np.asarray(mu_draws, dtype=float)
    if mu_draws.ndim == 3:
        if mu_draws.shape[1] != 1:
            raise ValueError(
                f"mu_draws has {mu_draws.shape[1]} components; "
                "effective_ledoit_wolf_shrinkage expects a single component. "
                "Slice with mu_draws[:, k, :] before calling."
            )
        mu_draws = mu_draws[:, 0, :]
    if mu_draws.ndim != 2:
        raise ValueError(f"mu_draws must be 2D (n_draws, kmax), got shape {mu_draws.shape}")
    if mu_draws.shape[0] < 2:
        raise ValueError("Need at least 2 posterior draws to fit Ledoit-Wolf")

    n_draws, kmax = mu_draws.shape
    X = logit(np.clip(mu_draws, eps, 1.0 - eps))

    delta = float(LedoitWolf().fit(X).shrinkage_)
    reg_eff = delta / (1.0 - delta) if delta < 1.0 else float("inf")

    return {
        "shrinkage": delta,
        "reg_eff":   reg_eff,
        "n_draws":   int(n_draws),
        "kmax":      int(kmax),
    }


def simple_reference_logit(beta: float, kmax: int, eps: float = 1e-8) -> np.ndarray:
    """Logit-scale simple contagion reference kernel.

    mu_sc(nu) = 1 - (1 - beta)^nu, nu = 1..kmax, returned on the logit scale.
    This is the best-fitting simple kernel reference once beta = beta_hat.
    """
    nu = np.arange(1, int(kmax) + 1, dtype=float)
    curve = 1.0 - (1.0 - float(beta)) ** nu
    return logit(np.clip(curve, eps, 1.0 - eps))


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
    z = solve_triangular(L, diff.T, lower=True)
    distances = np.sum(z ** 2, axis=0)

    erank = _erank_from_regularized(sigma)
    return distances, float(distances.mean()), erank