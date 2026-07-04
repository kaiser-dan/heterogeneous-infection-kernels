"""Data generating process (DGP). Takes structured params -> (M, N).

Example:
    {
        "graph": {"type": "sbm", "params": {"n": 256, "avg_degree": 18, "epsilon": 0.9, "delta": 0.0}},
        "gamma": 0.1, "rho": 0.125, "tmax": 1000, "rep": 0,
        "kernels":    [{"type": "simple", "beta": 0.01}, {"type": "simple", "beta": 0.05}],
        "assignment": {"mode": "block_flip", "omega": 0.1},
    }

A single-kernel DGP uses a one-element `kernels` list; `assignment` is then ignored.
"""

import sys
from pathlib import Path

import numpy as np

# assumes this file is at <repo_root>/workflow/dgp.py
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.simulation import (
    build_graph,
    build_kernel_bank,
    load_kernel,
    assign_kernels,
    build_initial_state,
    heterogeneous_contagion_process,
    get_activations,
)


def simulate_MN(dgp: dict):
    """Maps DGP params -> (M, N)."""
    rep = dgp["rep"]

    A, kmax = build_graph(dgp["graph"], seed=rep)
    n = A.shape[0]

    kernels_cfg = [
        {**k, "values": load_kernel(k["values_file"], kmax)}
        if k.get("type") == "nonparametric" and "values_file" in k
        else k
        for k in dgp["kernels"]
    ]
    bank      = build_kernel_bank(kernels_cfg, kmax=kmax)
    n_kernels = len(dgp["kernels"])
    if n_kernels == 1:
        labels = np.zeros(n, dtype=int) # everyone -> kernel 0
    else:
        labels = assign_kernels(
            n, dgp["assignment"], n_kernels=n_kernels, seed=rep)

    c  = np.stack([bank[labels[l]] for l in range(n)])
    x0 = build_initial_state(n, dgp["rho"], seed=rep)
    x  = heterogeneous_contagion_process(
        A, dgp["gamma"], c, x0,
        tmin=0, tmax=dgp["tmax"],
        random_seed=rep,        # deterministic given rep
    )
    M, N = get_activations(np.transpose(x), A, kmax)
    return M, N
