import os
from enum import Enum
from typing import Union

import cmdstanpy as csp

__all__ = ["Model", "fit_model"]


# Ordered chronologically by conception, values not otherwise meaningful
class Model(Enum):
    JGY = 1
    JGY_mod = 2
    NWL = 3
    KT = 4


def fit_model(
    data: dict,
    model: Model = Model.JGY_mod,
    num_chains: int = 1,
    num_samples: int = 100,
    adapt_delta: float = 0.8,
    warmups_per_sample: int = 4,
    seed: Union[None, int] = 37
) -> csp.CmdStanVB:
    # Validate data object
    assert "V" in data
    assert "kmax" in data
    assert "K" in data
    assert "M" in data
    assert "N" in data
    
    # Dispatch Stan model
    if model == Model.JGY:
        # Set filepaths
        raise NotImplementedError()

        # Validate model-specific (hyper)parameters
        assert "DEVIATION_SCALE" in data
    elif model == Model.JGY_mod:
        # Set filepaths
        stan_filepath  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "model_JGY-mod.stan")
        model_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "model_JGY-mod")

        # Validate model-specific (hyper)parameters
        assert "DEVIATION_SCALE" in data
        assert "REPLICATES" in data  # TODO: Remove predictive distribution sampling
    elif model == Model.NWL:
        # Set filepaths
        stan_filepath  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "model_nonlogit.stan")
        model_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "model_nonlogit")

        # Validate model-specific (hyper)parameters
        # TODO: Add hyperpriors on mean and strength of beta priors
        # assert "THETA" in data
        # assert "PHI" in data
    elif model == Model.KT:
        # Set filepaths
        stan_filepath  = os.path.join(os.path.abspath(__file__), "models", "model_KT.stan")
        model_filepath = os.path.join(os.path.abspath(__file__), "..", "build", "model_KT")

        # Validate model-specific (hyper)parameters
        # assert "DEVIATION_SCALE" in data
    else:
        raise ValueError("Model must be an instance of the 'Model' enum!")

    # Instantiate CmdStanPy model object
    stanmodel = csp.CmdStanModel(
        stan_file = stan_filepath,
        exe_file  = model_filepath,
        force_compile=False
    )

    # Fit model
    fit = stanmodel.sample(
        data          = data,
        seed          = seed,
        adapt_delta   = adapt_delta,
        chains        = num_chains,
        iter_sampling = num_samples,
        iter_warmup   = warmups_per_sample*num_samples,
        show_progress = False,
        show_console  = False
    )

    return fit