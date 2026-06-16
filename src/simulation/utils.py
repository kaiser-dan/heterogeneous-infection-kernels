from __future__ import annotations

import numpy as np

from .networks import er, sbm

__all__ = ["simple_kernel", "hill_kernel", "nonparametric_kernel", "build_kernel_bank", "build_graph", "assign_kernels", "build_initial_state"]

def simple_kernel(beta: float, kmax: int) -> np.ndarray:
    """Simple (independent cascade) kernel: P(ν) = 1 - (1 - β)^ν.

    Parameters
    ----------
    beta : float
        Per-contact infection probability in (0, 1).
    kmax : int
        Maximum number of infected neighbours.

    Returns
    -------
    (kmax,) array on the probability scale, ν = 1..kmax.
    """
    nu = np.arange(1, kmax + 1, dtype=float)
    return 1.0 - (1.0 - float(beta)) ** nu


def hill_kernel(p: float, h: float, tau: float, kmax: int) -> np.ndarray:
    """Hill-function (complex contagion) kernel.

    P(ν) = p · νʰ / (τʰ + νʰ)

    Sigmoid-shaped, saturating at p. Controls:
      p   — saturation ceiling
      h   — Hill coefficient (steepness; h > 1 for sigmoidal shape)
      tau — half-saturation point (exposure level where P = p/2)

    Parameters
    ----------
    p : float
        Saturation ceiling in (0, 1].
    h : float
        Hill coefficient (steepness), h > 0.
    tau : float
        Half-saturation exposure level, tau > 0.
    kmax : int
        Maximum number of infected neighbours.

    Returns
    -------
    (kmax,) array on the probability scale, ν = 1..kmax.
    """
    nu = np.arange(1, kmax + 1, dtype=float)
    return float(p) * (nu ** float(h)) / (float(tau) ** float(h) + nu ** float(h))


def nonparametric_kernel(values: list[float], kmax: int) -> np.ndarray:
    """Arbitrary kernel specified as an explicit list of probability values.

    Parameters
    ----------
    values : list of floats
        One probability value per exposure level, length >= kmax.
    kmax : int
        Maximum number of infected neighbours. Truncates values to this length.

    Returns
    -------
    (kmax,) array on the probability scale.
    """
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size < kmax:
        raise ValueError(
            f"Nonparametric kernel has {arr.size} values but kmax={kmax}."
        )
    return arr[:kmax]


def build_kernel_bank(kernels_cfg: list[dict], kmax: int) -> list[np.ndarray]:
    """Construct a list of kernel arrays from a config-style list of dicts.

    Each dict must have a 'type' key ('simple', 'hill', or 'nonparametric')
    and the corresponding parameters. Designed to accept the kernels list
    from config.yaml directly, but equally usable in notebooks:

        kernels = build_kernel_bank([
            {"type": "simple", "beta": 0.05},
            {"type": "hill",   "p": 0.8, "h": 2.0, "tau": 4.0},
        ], kmax=20)

    Parameters
    ----------
    kernels_cfg : list of dicts, one per kernel type.
    kmax : int
        Maximum exposure level for all kernels.

    Returns
    -------
    List of (kmax,) arrays, one per kernel.
    """
    bank = []
    for cfg in kernels_cfg:
        ktype = cfg.get("type")
        if ktype == "simple":
            bank.append(simple_kernel(beta=float(cfg["beta"]), kmax=kmax))
        elif ktype == "hill":
            bank.append(hill_kernel(
                p=float(cfg["p"]),
                h=float(cfg["h"]),
                tau=float(cfg["tau"]),
                kmax=kmax,
            ))
        elif ktype == "nonparametric":
            bank.append(nonparametric_kernel(values=cfg["values"], kmax=kmax))
        else:
            raise ValueError(
                f"Unsupported kernel type: '{ktype}'. Expected simple, hill, nonparametric."
            )
    return bank


def build_graph(graph_cfg: dict, seed: int) -> np.ndarray:
    """Build an adjacency matrix from a config-style graph dict.

    Supported types: 'sbm' (stochastic block model), 'er' (Erdős–Rényi).

    Example
    -------
        adj = build_graph({
            "type": "sbm",
            "params": {"n": 500, "avg_degree": 10, "epsilon": 0.05}
        }, seed=42)

    Parameters
    ----------
    graph_cfg : dict with keys 'type' and 'params'.
    seed : int

    Returns
    -------
    (n, n) integer adjacency matrix.
    """
    graph_type = graph_cfg.get("type", "sbm")
    params     = graph_cfg.get("params", {})

    if graph_type == "sbm":
        return np.asarray(sbm(
            n=int(params["n"]),
            k=float(params.get("avg_degree", params.get("k"))),
            epsilon=float(params["epsilon"]),
            delta=float(params.get("delta", 0.5)),
            seed=seed,
        ), dtype=int)

    if graph_type == "er":
        return np.asarray(er(
            n=int(params["n"]),
            p=params.get("p"),
            k=params.get("avg_degree"),
            seed=seed,
        ), dtype=int)

    raise ValueError(f"Unsupported graph type: '{graph_type}'. Expected sbm or er.")


def assign_kernels(
    n: int,
    graph_type: str,
    assignment_cfg: dict,
    n_kernels: int,
    seed: int,
) -> np.ndarray:
    """Assign one kernel index to each of n nodes.

    Modes
    -----
    block_flip : Split nodes into two equal halves by index, then randomly
                 flip assignments with probability omega. Requires SBM and
                 exactly 2 kernels. Useful for correlated community structure.
    iid        : Each node independently draws a kernel index from a
                 categorical distribution. Default: uniform over kernels.
    explicit   : Provide a labels array directly in assignment_cfg['labels'].

    Example
    -------
        labels = assign_kernels(
            n=500, graph_type="sbm",
            assignment_cfg={"mode": "block_flip", "omega": 0.1},
            n_kernels=2, seed=42,
        )

    Parameters
    ----------
    n : int
        Number of nodes.
    graph_type : str
        Graph type string, used to validate block_flip mode.
    assignment_cfg : dict
        Must contain 'mode' key plus mode-specific parameters.
    n_kernels : int
        Number of kernels in the bank.
    seed : int

    Returns
    -------
    (n,) integer array of kernel indices in [0, n_kernels).
    """
    mode = assignment_cfg.get("mode", "iid")
    rng  = np.random.default_rng(seed + 17)

    if mode == "block_flip":
        if n_kernels != 2:
            raise ValueError("block_flip requires exactly 2 kernels.")
        if graph_type != "sbm":
            raise ValueError("block_flip is only supported for SBM graphs.")
        omega  = float(assignment_cfg.get("omega", 0.0))
        labels = np.zeros(n, dtype=int)
        labels[n // 2:] = 1
        labels[rng.random(n) < omega] ^= 1
        return labels

    if mode == "iid":
        probs = assignment_cfg.get("probs")
        if probs is None:
            probs = np.ones(n_kernels, dtype=float) / n_kernels
        else:
            probs = np.asarray(probs, dtype=float).reshape(-1)
            if probs.size != n_kernels:
                raise ValueError("Length of assignment.probs must equal number of kernels.")
            probs = probs / probs.sum()
        return rng.choice(n_kernels, size=n, p=probs).astype(int)

    if mode == "explicit":
        labels = np.asarray(assignment_cfg.get("labels", []), dtype=int).reshape(-1)
        if labels.size != n:
            raise ValueError("Explicit assignment requires one label per node.")
        return labels

    raise ValueError(
        f"Unsupported assignment mode: '{mode}'. Expected block_flip, iid, explicit."
    )


def build_initial_state(n: int, initial_infected: int | float, seed: int) -> np.ndarray:
    """Build an initial infection state vector x0.

    Parameters
    ----------
    n : int
        Number of nodes.
    initial_infected : int or float
        Number of initially infected nodes (int), or fraction of n (float in (0,1]).
    seed : int

    Returns
    -------
    (n,) binary array with initial_infected randomly chosen nodes set to 1.
    """
    rng = np.random.default_rng(seed + 23)
    if isinstance(initial_infected, float) and initial_infected <= 1.0:
        n_init = max(1, int(round(initial_infected * n)))
    else:
        n_init = int(initial_infected)
    n_init = max(1, min(n_init, n))
    x0     = np.zeros(n, dtype=int)
    x0[rng.choice(n, size=n_init, replace=False)] = 1
    return x0
