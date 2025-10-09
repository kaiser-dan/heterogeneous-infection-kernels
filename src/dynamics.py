import numpy as np
from numpy import ndarray

__all__ = ["heterogeneous_contagion_process", "get_activations"]

# Shamelessly stolen from Nick at:
# https://github.com/nwlandry/heterogeneous-complex-network-reconstruction/blob/main/lcs/contagion.py
def heterogeneous_contagion_process(
    A, gamma, c, x0, tmin=0, tmax=20, dt=1, random_seed=None
):
    """A neighborhood-based contagion process on pairwise networks

    Parameters
    ----------
    A : Numpy ndarray
        The adjacency matrix
    gamma : float
        The healing rate
    c : Numpy ndarray
        A 2d vector of the contagion rates, where each row is a contagion function.
        Should be N x N.
    x0 : Numpy ndarray
        A 1D vector of the initial nodal states, either 0 or 1.
        Should be N x 1.
    tmin : int, optional
        The time at which to start the simulation, by default 0
    tmax : float, optional
        The time at which to terminate the simulation, by default 20
    dt : float, optional
        The time step, by default 1
    random_seed : int, optional
        The seed for the random process, by default None

    Returns
    -------
    Numpy ndarray
        The time series of the contagion process. Should be T x N.

    Raises
    ------
    Exception
        If adjacency matrix isn't square
    """
    if not isinstance(A, ndarray):
        A = A.todense()

    if A.dtype != "int":
        A = np.array(A, dtype=int)

    if random_seed is not None:
        np.random.seed(random_seed)

    n, m = np.shape(A)
    if n != m:
        raise Exception("Matrix must be square!")

    T = int((tmax - tmin) / dt)
    x = np.zeros((T, n), dtype=int)
    x[0] = x0.copy()

    for t in range(T - 1):
        u = np.random.random(n)
        nu = A @ x[t]
        for i in range(n):
            x[t + 1, i] = (1 - x[t, i]) * (u[i] <= c[i, nu[i]] * dt) + x[t, i] * (
                u[i] > gamma * dt
            )
    return x

def get_activations(states):