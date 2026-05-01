import numpy as np
import pytest
from src.models.inference import compute_held_out_loglik


class MockFit:
    def __init__(self, mu_vals, pi_vals):
        self._vars = {"mu": np.asarray(mu_vals, dtype=float),
                      "pi": np.asarray(pi_vals, dtype=float)}

    def stan_variable(self, name):
        return self._vars[name]


def test_compute_held_out_loglik_k1_known_value():
    # K=1, kmax=3, S=2 draws (both identical), V_held=1
    # mu[s=0, k=0, :] = [0.1, 0.2, 0.3] for both draws
    # pi[s=0, k=0] = 1.0 for both draws
    # M_held = [[1, 0, 0]], N_held = [[0, 1, 0]]
    # Expected per-draw log-lik: log(pi=1) + 1*log(0.1) + 1*log(1-0.2)
    #                           = log(0.1) + log(0.8) = log(0.08)
    # log_pred = logsumexp([log(0.08), log(0.08)]) - log(2) = log(0.08)
    # cv_score = log(0.08)
    mu = np.array([[[0.1, 0.2, 0.3]], [[0.1, 0.2, 0.3]]])  # (S=2, K=1, kmax=3)
    pi = np.array([[1.0], [1.0]])                           # (S=2, K=1)
    fit = MockFit(mu, pi)

    M_held = np.array([[1.0, 0.0, 0.0]])  # (V=1, kmax=3)
    N_held = np.array([[0.0, 1.0, 0.0]])  # (V=1, kmax=3)

    result = compute_held_out_loglik(M_held, N_held, fit)

    expected_cv_score = np.log(0.08)
    assert abs(result["cv_score"] - expected_cv_score) < 1e-9
    assert result["n_held_out"] == 1
    assert abs(result["mean_log_pred"] - expected_cv_score) < 1e-9


def test_compute_held_out_loglik_k2_mixture():
    # K=2, kmax=2, S=1, V_held=1
    # Component 0: mu=[0.5, 0.5], pi=0.5
    # Component 1: mu=[0.5, 0.5], pi=0.5
    # Since both components identical, result = log(0.5 * p^M * (1-p)^N + 0.5 * p^M * (1-p)^N)
    # = log(p^M * (1-p)^N)
    mu = np.array([[[0.5, 0.5], [0.5, 0.5]]])   # (S=1, K=2, kmax=2)
    pi = np.array([[0.5, 0.5]])                   # (S=1, K=2)
    fit = MockFit(mu, pi)

    M_held = np.array([[1.0, 0.0]])
    N_held = np.array([[0.0, 1.0]])

    result = compute_held_out_loglik(M_held, N_held, fit)
    # log(0.5^1 * 0.5^1) = log(0.25)
    assert abs(result["cv_score"] - np.log(0.25)) < 1e-9


def test_compute_held_out_loglik_empty_held_out():
    mu = np.array([[[0.1, 0.2]]])  # (S=1, K=1, kmax=2)
    pi = np.array([[1.0]])
    fit = MockFit(mu, pi)

    M_held = np.zeros((0, 2))
    N_held = np.zeros((0, 2))

    result = compute_held_out_loglik(M_held, N_held, fit)
    assert result["cv_score"] == 0.0
    assert result["n_held_out"] == 0
    assert np.isnan(result["mean_log_pred"])


def test_compute_held_out_loglik_k1_cmdstanpy_squeeze():
    """CmdStanPy returns mu as (S, kmax) and pi as (S,) when K=1."""
    mu = np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])  # (S=2, kmax=3), ndim=2
    pi = np.array([1.0, 1.0])                            # (S=2,), ndim=1
    fit = MockFit(mu, pi)

    M_held = np.array([[1.0, 0.0, 0.0]])
    N_held = np.array([[0.0, 1.0, 0.0]])
    result = compute_held_out_loglik(M_held, N_held, fit)
    # Same expected value as the 3D K=1 test
    assert abs(result["cv_score"] - np.log(0.08)) < 1e-9
