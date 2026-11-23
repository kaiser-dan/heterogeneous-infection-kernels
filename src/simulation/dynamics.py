import sys

import numpy as np
from numpy import ndarray

__all__ = ["heterogeneous_contagion_process", "get_activations", "build_initial_state"]

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
    # if n != m:
    #     raise Exception("Matrix must be square!")

    T = int((tmax - tmin) / dt)
    x = np.zeros((T, n), dtype=int)
    x[0] = x0.copy()

    rows = np.arange(n)
    for t in range(T - 1):
        u = np.random.random(n)
        nu = A @ x[t]
        # Infection probability c_i(nu); column j of c holds nu = j + 1, and c(0) = 0
        p_inf = np.where(nu > 0, c[rows, np.maximum(nu - 1, 0)], 0.0)

        # Susceptible node infected wp c_i(nu) * dt
        infect = (x[t] == 0) & (u <= p_inf * dt)
        # Infected node recovers wp gamma * dt
        stay_infected = (x[t] == 1) & (u > gamma * dt)

        x[t + 1] = (infect | stay_infected).astype(int)
    return x

def get_activations(states: ndarray, adjacency: ndarray, kmax: int):
    """Count infection (M) and non-infection (N) events by node and exposure level.

    Parameters
    ----------
    states : (N, T) array of 0/1 node states (transpose of the process output).
    adjacency : (N, N) adjacency matrix.
    kmax : int
        Number of exposure columns; column j counts events with nu = j + 1.

    Only observed transitions t -> t+1 (t = 0, ..., T-2) are counted.
    """
    states = np.asarray(states, dtype=int)
    n_nodes, T = states.shape

    # Number of infected neighbors at each time step
    nu = np.asarray(adjacency @ states, dtype=int)

    # Observation = susceptible at t (t <= T-2) with at least one infected neighbor
    observed = (states[:, :-1] == 0) & (nu[:, :-1] >= 1)
    infected_next = states[:, 1:] == 1

    if observed.any() and nu[:, :-1][observed].max() > kmax:
        raise ValueError(f"Exposure level {nu[:, :-1][observed].max()} exceeds kmax={kmax}.")

    infection_events = np.zeros((n_nodes, kmax), dtype=int)
    noninfection_events = np.zeros((n_nodes, kmax), dtype=int)
    node_idx = np.broadcast_to(np.arange(n_nodes)[:, None], observed.shape)

    inf = observed & infected_next
    non = observed & ~infected_next
    np.add.at(infection_events, (node_idx[inf], nu[:, :-1][inf] - 1), 1)
    np.add.at(noninfection_events, (node_idx[non], nu[:, :-1][non] - 1), 1)

    return infection_events, noninfection_events

def build_initial_state(n: int, initial_infected: int | float, seed=None) -> np.ndarray:
    """Build an initial infection state vector x0.

    Parameters
    ----------
    n : int
        Number of nodes.
    initial_infected : int or float
        Number of initially infected nodes (int), or fraction of n (float in (0,1]).
    seed : int, optional

    Returns
    -------
    (n,) binary array with initial_infected randomly chosen nodes set to 1.
    """
    rng = np.random.default_rng(seed)
    
    if isinstance(initial_infected, float) and initial_infected <= 1.0:
        n_init = max(1, int(round(initial_infected * n)))
    else:
        n_init = int(initial_infected)
        
    n_init = max(1, min(n_init, n))
    x0     = np.zeros(n, dtype=int)
    x0[rng.choice(n, size=n_init, replace=False)] = 1
    
    return x0
