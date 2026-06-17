from __future__ import annotations

import numpy as np

__all__ = ["simple_kernel", "hill_kernel", "nonparametric_kernel", "build_kernel_bank", "assign_kernels"]

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

def sigmoid_kernel(rho_l: float, rho_h: float, w: float, k0: float, kmax: int) -> np.ndarray:
    """Sigmoidal (complex contagion) kernel.

    Parameters
    ----------
    rho_l : float
        Baseline infection rate at low exposure.
    rho_h : float
        Saturation infection rate at high exposure.
    w : float
        Steepness of the sigmoid transition.
    k0 : float
        Half-saturation exposure level.
    kmax : int
        Maximum number of infected neighbors.

    Returns
    -------
    np.ndarray of shape (kmax,)
        Infection rate at each exposure level nu = 1, ..., kmax.
    """
    nu = np.arange(1, kmax + 1, dtype=float)
    return rho_l + (rho_h - rho_l) / (1.0 + np.exp(-w * (nu - k0)))

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

    Each dict must have a 'type' key ('simple', 'sigmoid', 'hill', or 'nonparametric')
    and the corresponding parameters. 
    
    Example
    -------
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

def assign_kernels(
    n: int,
    assignment_cfg: dict,
    n_kernels: int,
    seed: int,
) -> np.ndarray:
    """Assign one kernel index to each of n nodes.

    Modes
    -----
    block_flip : Split nodes into two communities, then randomly flip assignments
                 with probability omega. Requires exactly 2 kernels.
                 Parameters:
                 omega       : float, flip probability (default 0.0)
                 communities : (n,) int array with exactly 2 unique values indicating
                               community membership. Values are remapped to 0/1 internally.

    iid        : Each node independently draws a kernel index from a categorical
                 distribution.
                 Parameters:
                 probs : (n_kernels,) float array of kernel probabilities;
                         if omitted, defaults to uniform.

    explicit   : Provide a labels array directly.
                 Parameters:
                 labels : (n,) int array of kernel indices in [0, n_kernels).

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
        
        communities = assignment_cfg.get("communities")
        if communities is not None:
            labels = np.asarray(communities, dtype=int).reshape(-1)
            if labels.size != n:
                raise ValueError("communities must have one entry per node.")
            unique = np.unique(labels)
            if unique.size != 2:
                raise ValueError("communities must contain exactly 2 unique values.")
            labels = (labels == unique[1]).astype(int)
        else:
            labels = np.zeros(n, dtype=int)
            labels[n // 2:] = 1
        omega = float(assignment_cfg.get("omega", 0.0))
        labels = labels.copy()
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