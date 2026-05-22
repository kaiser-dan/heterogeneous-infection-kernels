from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MalahboisResult:
    """Mahalanobis-based complexity outputs for one posterior."""

    total_distance: np.ndarray
    centered_distance: np.ndarray
    non_centrality: float
    mode_non_centrality: float
    reference_kernel: np.ndarray
    mode_reference_kernel: np.ndarray
    posterior_mean: np.ndarray
    posterior_mode: np.ndarray
    covariance: np.ndarray
    mode_covariance: np.ndarray
    rho_mode: float


def simple_contagion_reference(rho: float, k: int) -> np.ndarray:
    """Simple contagion kernel c(nu) = 1 - (1 - rho)^nu for nu=1..k."""
    nu = np.arange(1, k + 1, dtype=float)
    return 1.0 - (1.0 - float(rho)) ** nu


def _regularized_covariance(samples: np.ndarray, jitter: float) -> np.ndarray:
    covariance = np.cov(samples, rowvar=False)
    covariance = np.atleast_2d(np.asarray(covariance, dtype=float))
    dim = covariance.shape[0]
    return covariance + (jitter * np.eye(dim, dtype=float))


def _whiten_samples(samples: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    chol = np.linalg.cholesky(covariance)
    return np.linalg.solve(chol, np.asarray(samples, dtype=float).T).T


def _pairwise_squared_distances(samples: np.ndarray) -> np.ndarray:
    sq_norms = np.sum(samples * samples, axis=1, keepdims=True)
    distances = sq_norms + sq_norms.T - (2.0 * samples @ samples.T)
    return np.maximum(distances, 0.0)


def _resolve_neighbor_count(n_samples: int, dimension: int) -> int:
    if n_samples < 3:
        return 1
    return min(max(int(np.sqrt(n_samples)), dimension + 2), n_samples - 1)


def _knn_mode_index(
    samples: np.ndarray,
    covariance: np.ndarray,
    neighbor_count: int,
) -> int:
    whitened = _whiten_samples(samples, covariance)
    distances = _pairwise_squared_distances(whitened)
    np.fill_diagonal(distances, np.inf)
    kth_radius = np.partition(distances, neighbor_count - 1, axis=1)[:, neighbor_count - 1]
    return int(np.argmin(kth_radius))


def _mode_neighborhood_indices(
    samples: np.ndarray,
    global_covariance: np.ndarray,
    mode_index: int,
    neighbor_count: int,
) -> np.ndarray:
    if samples.shape[0] < 3:
        return np.arange(samples.shape[0], dtype=int)

    local_sample_count = min(
        max(neighbor_count + 1, samples.shape[1] + 2),
        samples.shape[0],
    )
    if local_sample_count < 2:
        return np.arange(samples.shape[0], dtype=int)

    whitened = _whiten_samples(samples, global_covariance)
    mode_delta = whitened - whitened[mode_index]
    distances = np.einsum("ij,ij->i", mode_delta, mode_delta)
    return np.argpartition(distances, local_sample_count - 1)[:local_sample_count]


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
    covariance = _regularized_covariance(mu, jitter=jitter)
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

    mu_neighbor_count = _resolve_neighbor_count(n_samples=n_samples, dimension=k)
    mu_mode_index = _knn_mode_index(mu, covariance=covariance, neighbor_count=mu_neighbor_count)
    mu_neighbor_idx = _mode_neighborhood_indices(
        samples=mu,
        global_covariance=covariance,
        mode_index=mu_mode_index,
        neighbor_count=mu_neighbor_count,
    )
    posterior_mode = mu[mu_neighbor_idx].mean(axis=0)
    mode_covariance = covariance

    rho_matrix = rho[:, None]
    rho_covariance = _regularized_covariance(rho_matrix, jitter=jitter)
    rho_neighbor_count = _resolve_neighbor_count(n_samples=rho.size, dimension=1)
    rho_mode_index = _knn_mode_index(
        rho_matrix,
        covariance=rho_covariance,
        neighbor_count=rho_neighbor_count,
    )
    rho_neighbor_idx = _mode_neighborhood_indices(
        samples=rho_matrix,
        global_covariance=rho_covariance,
        mode_index=rho_mode_index,
        neighbor_count=rho_neighbor_count,
    )
    rho_mode = float(rho[rho_neighbor_idx].mean())
    mu_sc_mode = simple_contagion_reference(rho_mode, k)

    z_mode_offset = np.linalg.solve(chol, posterior_mode - mu_sc_mode)
    mode_non_centrality = float(np.dot(z_mode_offset, z_mode_offset))

    if total_distance.shape != (n_samples,) or centered_distance.shape != (n_samples,):
        raise RuntimeError("Unexpected Mahalanobis output shape.")

    return MalahboisResult(
        total_distance=total_distance,
        centered_distance=centered_distance,
        non_centrality=non_centrality,
        mode_non_centrality=mode_non_centrality,
        reference_kernel=mu_sc,
        mode_reference_kernel=mu_sc_mode,
        posterior_mean=mu_bar,
        posterior_mode=posterior_mode,
        covariance=covariance,
        mode_covariance=mode_covariance,
        rho_mode=rho_mode,
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
