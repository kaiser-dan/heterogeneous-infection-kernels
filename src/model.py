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
    adapt_delta: float = 0.9,
    warmups_per_sample: int = 4,
    seed: Union[None, int] = 37
) -> csp.CmdStanVB:
    # Dispatch Stan model
    if model == Model.JGY:
        raise NotImplementedError()
    elif model == Model.JGY_mod:
        stan_filepath  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "model_JGY-mod.stan")
        model_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "model_JGY-mod")
    elif model == Model.NWL:
        stan_filepath  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "model_nonlogit.stan")
        model_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "model_nonlogit")
    elif model == Model.KT:
        stan_filepath  = os.path.join(os.path.abspath(__file__), "models", "model_KT.stan")
        model_filepath = os.path.join(os.path.abspath(__file__), "..", "build", "model_KT")
    else:
        raise ValueError("Model must be an instance of the 'Model' enum!")

    # Instantiate CmdStanPy model object
    stanmodel = csp.CmdStanModel(
        stan_file = stan_filepath,
        exe_file  = model_filepath
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