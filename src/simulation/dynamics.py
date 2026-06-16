import sys

import numpy as np
from numpy import ndarray

from .._logger import logger

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
    # if n != m:
    #     raise Exception("Matrix must be square!")

    T = int((tmax - tmin) / dt)
    x = np.zeros((T, n), dtype=int)
    x[0] = x0.copy()

    for t in range(T - 1):
        u = np.random.random(n)
        nu = A @ x[t]
        for i in range(n):
            if nu[i] > 0:
                _a = (1 - x[t, i]) * (u[i] <= c[i, nu[i]-1] * dt) 
                _b = x[t, i] * (u[i] > gamma * dt)
            else:
                _a, _b = 0, 0    
            x[t + 1, i] = _a + _b

            # x[t + 1, i] = (1 - x[t, i]) * (u[i] <= c[i, nu[i]] * dt) + x[t, i] * (
            #     u[i] > gamma * dt
            # )
    return x

def get_activations(states: ndarray, adjacency: ndarray, kmax: int):
    # Unpack number of individuals and epidemic time scale
    N, _ = states.shape

    # Get number of infected neighbors at each timestep
    nu = np.array(adjacency @ states, dtype=int)

    # Retrieve timesteps where infection events are possible
    possible_infection_events = [
        np.where(1-states[i, :])  # 1 minus because we want when succeptibility
        for i in range(N)
    ]

    # Compute infection events
    infection_events = np.zeros((N, kmax))
    noninfection_events = np.zeros((N, kmax))
    for i in range(N):
        # Unpack times when this individual (i) is succeptible
        possible_infection_events_individual = set(possible_infection_events[i][0])

        # Count what happens at each time this node is succeptible (infection event or not)
        for t in possible_infection_events_individual:
            # Adjust for 0-indexing, i.e., first column (index 0) represents 1 infected neighbor
            nu_i_t = nu[i][t]-1
            
            # If no infected neighbors, index adjustment becomes negative and we have no update to event counts
            if nu_i_t < 0:
                continue

            # Infection DOES NOT occur if S at t and at t+1
            if t+1 in possible_infection_events_individual:  
                noninfection_events[i][nu_i_t] += 1
                logger.trace(f"Failed iInfection event for node {i} at time {t} with {nu_i_t} neighbors")

            # Infection DOES occur if S at t but not at t+1
            else:
                infection_events[i][nu_i_t] += 1
                logger.trace(f"Infection event for node {i} at time {t} with {nu_i_t} neighbors")

    # Ensure integer type
    infection_events = np.array(infection_events, dtype=int)
    noninfection_events = np.array(noninfection_events, dtype=int)

    return infection_events, noninfection_events