"""Gauss-Hermite quadrature helper for the marginalized nonparametric-EB model."""

import numpy as np


def gauss_hermite(G: int) -> tuple[np.ndarray, np.ndarray]:
    """Physicists' Gauss-Hermite nodes/weights: INT exp(-x^2) f(x) dx ~= sum_g w_g f(x_g)."""
    nodes, weights = np.polynomial.hermite.hermgauss(int(G))
    return nodes, weights
