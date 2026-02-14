from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MalahboisResult:
    """Mahalanobis-based complexity outputs for one posterior."""

    total_distance: np.ndarray
    centered_distance: np.ndarray
    non_centrality: float
    reference_kernel: np.ndarray
    posterior_mean: np.ndarray
    covariance: np.ndarray


def simple_contagion_reference(rho: float, k: int) -> np.ndarray:
    """Simple contagion kernel c(nu) = 1 - (1 - rho)^nu for nu=1..k."""
    nu = np.arange(1, k + 1, dtype=float)
    return 1.0 - (1.0 - float(rho)) ** nu


def compute_malahbois_complexity(
    mu_samples: np.ndarray,
    rho_samples: np.ndarray,
    jitter: float = 1e-8,
) -> MalahboisResult:
    """
    Compute decomposed Mahalanobis complexity for posterior kernel samples.

    Parameters
    ----------
    mu_samples
        Posterior samples with shape (n_samples, k).
    rho_samples
        Simple-contagion rho posterior samples with shape (n_samples,) or (m,).
    jitter
        Small diagonal regularizer added to covariance for stable inversion.
    """
    mu = np.asarray(mu_samples, dtype=float)
    rho = np.asarray(rho_samples, dtype=float).reshape(-1)

    if mu.ndim != 2:
        raise ValueError("mu_samples must be a 2D array of shape (n_samples, k).")
    if mu.shape[0] < 2:
        raise ValueError("mu_samples must contain at least 2 samples.")
    if rho.size == 0:
        raise ValueError("rho_samples must contain at least one sample.")
    if jitter <= 0:
        raise ValueError("jitter must be positive.")

    n_samples, k = mu.shape
    mu_bar = mu.mean(axis=0)
    covariance = np.cov(mu, rowvar=False)
    covariance = covariance + (jitter * np.eye(k, dtype=float))
    chol = np.linalg.cholesky(covariance)

    rho_hat = float(rho.mean())
    mu_sc = simple_contagion_reference(rho_hat, k)

    delta_total = mu - mu_sc
    z_total = np.linalg.solve(chol, delta_total.T).T
    total_distance = np.einsum("ij,ij->i", z_total, z_total)

    delta_centered = mu - mu_bar
    z_centered = np.linalg.solve(chol, delta_centered.T).T
    centered_distance = np.einsum("ij,ij->i", z_centered, z_centered)

    z_offset = np.linalg.solve(chol, mu_bar - mu_sc)
    non_centrality = float(np.dot(z_offset, z_offset))

    if total_distance.shape != (n_samples,) or centered_distance.shape != (n_samples,):
        raise RuntimeError("Unexpected Mahalanobis output shape.")

    return MalahboisResult(
        total_distance=total_distance,
        centered_distance=centered_distance,
        non_centrality=non_centrality,
        reference_kernel=mu_sc,
        posterior_mean=mu_bar,
        covariance=covariance,
    )


def compute_mahalanobis_decomposed(
    mu_samples: np.ndarray,
    rho_samples: np.ndarray,
    jitter: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Backward-compatible helper returning the same tuple as notebook code.
    """
    result = compute_malahbois_complexity(mu_samples, rho_samples, jitter=jitter)
    return (
        result.total_distance,
        result.centered_distance,
        result.non_centrality,
        result.reference_kernel,
    )
