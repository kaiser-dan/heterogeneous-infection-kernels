import numpy as np
import pandas as pd

from src.models.inference import generate_stan_matrices_twitter


def test_survival_overflow_is_right_censored_for_adopters_k_metric():
    df = pd.DataFrame(
        [
            {"user_id": "u_overflow", "A_array": np.array([25]), "adopted": True},
            {"user_id": "u_exact", "A_array": np.array([20]), "adopted": True},
        ]
    )

    M, N, _ = generate_stan_matrices_twitter(
        summary_df=df,
        q=1.0,
        metric="k",
        kmax=20,
        mn_mode="survival",
    )

    # Overflow adopter: event is beyond observed window => right-censored at kmax.
    assert M[0, 19] == 0.0
    assert N[0, 19] == 1.0

    # Exact-level adopter: observed event at kmax.
    assert M[1, 19] == 1.0
    assert N[1, 19] == 0.0


def test_survival_overflow_is_right_censored_for_adopters_kappa_metric():
    df = pd.DataFrame(
        [
            {"user_id": "u_overflow", "A_array": np.ones(30, dtype=int), "adopted": True},
            {"user_id": "u_exact", "A_array": np.ones(20, dtype=int), "adopted": True},
        ]
    )

    M, N, _ = generate_stan_matrices_twitter(
        summary_df=df,
        q=1.0,
        metric="kappa",
        kmax=20,
        mn_mode="survival",
    )

    # Overflow adopter: event is beyond observed window => right-censored at kmax.
    assert M[0, 19] == 0.0
    assert N[0, 19] == 1.0

    # Exact-level adopter: observed event at kmax.
    assert M[1, 19] == 1.0
    assert N[1, 19] == 0.0
