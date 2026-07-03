from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["simple_kernel", "hill_kernel", "nonparametric_kernel", "load_kernel", "build_kernel_bank", "assign_kernels"]

def simple_kernel(beta: float, kmax: int) -> np.ndarray:
    """Simple (independent cascade) kernel: c(ν) = 1 - (1 - β)^ν.

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

    c(ν) = p · νʰ / (τʰ + νʰ)

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


def power_kernel(lam: float, alpha: float, kmax: int) -> np.ndarray:
    """Sublinear/superlinear (complex contagion) kernel.

    c(ν) = 1 - exp(-lam * ν ^ alpha)

    Sublinear when alpha < 1, superlinear when alpha > 1.

    Parameters
    ----------
    lam : float
        Baseline cumulative hazard from a single infected neighbor.
        Single-exposure infection probability is c(1) = 1 - exp(-lam). 
    alpha : float
        Exposure-scaling exponent. alpha < 1 gives sublinear contagion, 
        alpha = 1 gives the standard simple-contagion kernel,
        and alpha > 1 gives superlinear contagion.
    kmax : int
        Maximum number of infected neighbours.

    Returns
    -------
    (kmax,) array on the probability scale, ν = 1..kmax.
    """
    nu = np.arange(1, kmax + 1, dtype=float)
    return 1.0 - np.exp(-lam * nu**alpha)


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


def load_kernel(path: str | Path, kmax: int) -> np.ndarray:
    """Load a nonparametric kernel's values from a text file, then build it.

    Parameters
    ----------
    path : str or Path
        Text file of whitespace/newline-separated floats, one per exposure
        level, length >= kmax.
    kmax : int
        Maximum number of infected neighbours. Truncates values to this length.

    Returns
    -------
    (kmax,) array on the probability scale.
    """
    values = np.loadtxt(path, dtype=float)
    return nonparametric_kernel(values, kmax=kmax)


def build_kernel_bank(kernels_cfg: list[dict], kmax: int) -> list[np.ndarray]:
    """Construct a list of kernel arrays from a config-style list of dicts.

    Each dict must have a 'type' key ('simple', 'hill', 'power', or 'nonparametric')
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
            
        elif ktype == "power":
            bank.append(power_kernel(
                lam=float(cfg["lam"]),
                alpha=float(cfg["alpha"]),
                kmax=kmax
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
    seed=None,
) -> np.ndarray:
    """Assign one kernel index to each of n nodes.

    Modes
    -----
    block_flip : Community-aligned assignment with flip probability omega.
                 Pr(kernel k | block b) = 1-(B-1)*omega if k == b, else omega.
                 omega = 0 is perfect alignment; omega = 1/B is uniform random.
                 omega       : float in [0, 1/n_kernels], default 0.0
                 communities : (n,) int array; remapped to 0..n_kernels-1.
                               If omitted, defaults to contiguous block assignment.

    iid        : Each node draws independently from a categorical distribution.
                 probs : (n_kernels,) weights; default uniform.

    explicit   : Pass labels directly.
                 labels : (n,) int array of kernel indices in [0, n_kernels).

    Example
    -------
        labels = assign_kernels(
            n=500,
            assignment_cfg={"mode": "block_flip", "omega": 0.9, "communities": communities},
            n_kernels=3, seed=42,
        )

    Parameters
    ----------
    n : int
    assignment_cfg : dict with 'mode' key plus mode-specific parameters.
    n_kernels : int
    seed : int, optional

    Returns
    -------
    (n,) integer array of kernel indices in [0, n_kernels).
    """
    mode = assignment_cfg.get("mode", "iid")
    rng  = np.random.default_rng(seed)

    if mode == "block_flip":
        communities = assignment_cfg.get("communities")
        omega = float(assignment_cfg.get("omega", 0.0))

        if communities is not None:
            labels = np.asarray(communities, dtype=int).reshape(-1)
            if labels.size != n:
                raise ValueError("communities must have one entry per node.")
            unique = np.unique(labels)
            if unique.size != n_kernels:
                raise ValueError(
                    f"communities has {unique.size} unique values but n_kernels={n_kernels}."
                )
            community_idx = np.searchsorted(unique, labels)
        else:
            block_sizes = [n // n_kernels] * n_kernels
            block_sizes[-1] += n - sum(block_sizes)
            community_idx = np.repeat(np.arange(n_kernels), block_sizes)

        if not (0.0 <= omega <= 1.0 / n_kernels):
            raise ValueError(
                f"omega must be in [0, 1/n_kernels] = [0, {1.0/n_kernels:.4f}], got {omega}."
            )

        p_other = omega
        p_same  = 1.0 - (n_kernels - 1) * omega
        kernel_labels = np.empty(n, dtype=int)
        for i in range(n):
            b = community_idx[i]
            probs = np.full(n_kernels, p_other)
            probs[b] = p_same
            kernel_labels[i] = rng.choice(n_kernels, p=probs)
        
        return kernel_labels

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