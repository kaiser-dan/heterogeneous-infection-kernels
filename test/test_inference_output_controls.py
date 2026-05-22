"""
Tests that inference-side Twitter M/N construction matches the current
data-processing row encoding for the same exposure distributions.
"""

import numpy as np
import pandas as pd

from src.data_processing.process_twitter import _mn_rows_from_prob_dist
from src.data_processing.utils import compute_binomial_distribution, compute_exposure_probabilities_fast
from src.models.inference import (
    convert_incidence_to_adopters_only_survival,
    generate_stan_matrices_twitter,
)


SUMMARY_DF = pd.DataFrame(
    [
        {"user_id": "u1", "A_array": np.array([2, 1]), "adopted": True},
        {"user_id": "u2", "A_array": np.array([1, 1]), "adopted": False},
        {"user_id": "u3", "A_array": np.array([3]), "adopted": True},
    ]
)

Q = 0.20
KMAX = 6


def _expected_rows(metric: str, mn_mode: str):
    expected_m = []
    expected_n = []

    for row in SUMMARY_DF.itertuples():
        if metric == "k":
            p_dist = compute_binomial_distribution(int(np.sum(row.A_array)), Q)
        else:
            p_dist = compute_exposure_probabilities_fast(list(row.A_array), Q)

        m_row, n_row = _mn_rows_from_prob_dist(
            p_dist,
            adopted=bool(row.adopted),
            kmax=KMAX,
            mn_mode=mn_mode,
        )
        expected_m.append(m_row)
        expected_n.append(n_row)

    return np.vstack(expected_m), np.vstack(expected_n)


def test_generate_stan_matrices_twitter_matches_processing_for_k():
    for mn_mode in ["incidence", "survival"]:
        M, N, row_weights = generate_stan_matrices_twitter(
            summary_df=SUMMARY_DF,
            q=Q,
            metric="k",
            kmax=KMAX,
            mn_mode=mn_mode,
        )
        expected_M, expected_N = _expected_rows("k", mn_mode)
        np.testing.assert_allclose(M, expected_M, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(N, expected_N, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(row_weights, np.ones(len(SUMMARY_DF)))


def test_generate_stan_matrices_twitter_matches_processing_for_kappa():
    for mn_mode in ["incidence", "survival"]:
        M, N, row_weights = generate_stan_matrices_twitter(
            summary_df=SUMMARY_DF,
            q=Q,
            metric="kappa",
            kmax=KMAX,
            mn_mode=mn_mode,
        )
        expected_M, expected_N = _expected_rows("kappa", mn_mode)
        np.testing.assert_allclose(M, expected_M, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(N, expected_N, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(row_weights, np.ones(len(SUMMARY_DF)))


def test_convert_incidence_to_adopters_only_survival_drops_non_adopters_and_marks_pre_event_risk():
    M_inc = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],  # adopter at bin 3
            [0.0, 0.0, 0.0, 0.0],  # non-adopter, should drop
            [0.0, 1.0, 0.0, 0.0],  # adopter at bin 2
        ]
    )

    M_surv, N_surv, adopted_mask = convert_incidence_to_adopters_only_survival(M_inc)

    np.testing.assert_array_equal(adopted_mask, np.array([True, False, True]))
    np.testing.assert_allclose(
        M_surv,
        np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        ),
    )
    np.testing.assert_allclose(
        N_surv,
        np.array(
            [
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        ),
    )


def test_convert_incidence_to_adopters_only_survival_preserves_fractional_tail_mass():
    M_inc = np.array([[0.2, 0.3, 0.5, 0.0]])

    M_surv, N_surv, adopted_mask = convert_incidence_to_adopters_only_survival(M_inc)

    np.testing.assert_array_equal(adopted_mask, np.array([True]))
    np.testing.assert_allclose(M_surv, M_inc)
    np.testing.assert_allclose(N_surv, np.array([[0.8, 0.5, 0.0, 0.0]]))
