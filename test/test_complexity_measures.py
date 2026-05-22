import numpy as np
import pytest

from src.complexity_measures.malahbois import (
    compute_mahalanobis_decomposed,
    compute_malahbois_complexity,
    simple_contagion_reference,
)
from workflow.scripts.analysis.run_ppc_nested_refit import stat_from_malahbois


def _draw_cluster(
    rng: np.random.Generator,
    center: np.ndarray,
    size: int,
    scale: float,
) -> np.ndarray:
    return center + rng.normal(scale=scale, size=(size, center.size))


def test_mode_non_centrality_tracks_dominant_mode_in_bimodal_posterior():
    rng = np.random.default_rng(1234)
    dominant_rho = 0.18
    dominant_center = simple_contagion_reference(dominant_rho, 4)
    secondary_center = np.array([0.64, 0.81, 0.9, 0.95], dtype=float)

    mu_samples = np.vstack(
        [
            _draw_cluster(rng, dominant_center, size=120, scale=0.015),
            _draw_cluster(rng, secondary_center, size=30, scale=0.015),
        ]
    )
    rho_samples = np.concatenate(
        [
            rng.normal(loc=dominant_rho, scale=0.01, size=120),
            rng.normal(loc=0.58, scale=0.01, size=30),
        ]
    )

    result = compute_malahbois_complexity(mu_samples=mu_samples, rho_samples=rho_samples)

    assert result.mode_non_centrality < result.non_centrality
    assert np.linalg.norm(result.posterior_mode - dominant_center) < np.linalg.norm(
        result.posterior_mean - dominant_center
    )
    assert abs(result.rho_mode - dominant_rho) < abs(float(rho_samples.mean()) - dominant_rho)
    assert stat_from_malahbois(mu_samples, rho_samples, "mahal_mode_non_centrality") == pytest.approx(
        result.mode_non_centrality
    )


def test_mode_non_centrality_stays_close_for_unimodal_posterior():
    rng = np.random.default_rng(2026)
    rho_true = 0.27
    center = simple_contagion_reference(rho_true, 5)

    mu_samples = _draw_cluster(rng, center, size=180, scale=0.02)
    rho_samples = rng.normal(loc=rho_true, scale=0.015, size=180)

    result = compute_malahbois_complexity(mu_samples=mu_samples, rho_samples=rho_samples)
    total_distance, centered_distance, non_centrality, reference_kernel = compute_mahalanobis_decomposed(
        mu_samples=mu_samples,
        rho_samples=rho_samples,
    )

    assert np.linalg.norm(result.posterior_mode - result.posterior_mean) < 0.08
    assert result.non_centrality < 0.5
    assert result.mode_non_centrality < 0.5
    assert result.mode_non_centrality / result.non_centrality < 4.0
    np.testing.assert_allclose(total_distance, result.total_distance)
    np.testing.assert_allclose(centered_distance, result.centered_distance)
    assert non_centrality == pytest.approx(result.non_centrality)
    np.testing.assert_allclose(reference_kernel, result.reference_kernel)
