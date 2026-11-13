"""Snakemake script: K=1 fit -> matched single-kernel truth (rule matched_kernel).

Input: fit .npz written by fit_model.py, with `draws_mu` of shape (S, 1, kmax) or (S, kmax).
Output: posterior mean of mu on probability scale, one value per exposure level nu = 1..kmax, 
as a text file readable by src.simulation.load_kernel.
"""

import numpy as np

# load posterior draws of mu
draws = np.asarray(np.load(snakemake.input[0])["draws_mu"], dtype=float)
if draws.ndim == 3:
    if draws.shape[1] != 1:
        raise ValueError(f"expected a K=1 fit, got draws_mu with shape {draws.shape}")
    draws = draws[:, 0, :]
if draws.ndim != 2:
    raise ValueError(f"unexpected draws_mu shape {draws.shape}")

kernel = draws.mean(axis=0)
if not np.all(np.isfinite(kernel)) or kernel.min() < 0.0 or kernel.max() > 1.0:
    raise ValueError(f"matched kernel is not a valid probability vector: {kernel}")

np.savetxt(snakemake.output[0], kernel)
