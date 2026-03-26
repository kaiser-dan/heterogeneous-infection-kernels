import networkx as nx
import numpy as np

__all__ = ["sbm", "er", "zkc"]


# Shamelessly stolen from Nick at:
# https://github.com/nwlandry/heterogeneous-complex-network-reconstruction/blob/main/lcs/generative.py
def sbm(n, k, epsilon, seed=None):
    """
    Generates a Stochastic Block Model (SBM) graph.

    Parameters
    ----------
    n : int
        The number of nodes in the graph.
    k : int
        The average degree of each node.
    epsilon : float
        The parameter controlling the ratio of inter- to intra-community edges.
    seed : int, optional
        The seed for the random number generator. Defaults to None.

    Returns
    -------
    numpy.ndarray
        The adjacency matrix of the generated SBM graph.

    Raises
    ------
    None

    """
    p = k / (n - 1)
    # ratio of inter- to intra-community edges
    p_in = (1 + epsilon) * p
    p_out = (1 - epsilon) * p
    G = nx.planted_partition_graph(2, int(n / 2), p_in, p_out, seed=seed)
    G.add_nodes_from(range(n))
    return np.asarray(nx.adjacency_matrix(G).todense(), dtype=int)


def er(n, p=None, k=None, seed=None):
    """
    Generate an Erdos-Renyi graph adjacency matrix.

    Provide exactly one of `p` or `k` (average degree).
    """
    if (p is None) == (k is None):
        raise ValueError("Provide exactly one of p or k.")
    if k is not None:
        if n <= 1:
            raise ValueError("n must be > 1 when specifying k.")
        p = float(k) / float(n - 1)
    G = nx.erdos_renyi_graph(n=n, p=float(p), seed=seed)
    G.add_nodes_from(range(n))
    return np.asarray(nx.adjacency_matrix(G).todense(), dtype=int)


def zkc():
    """
    Generate Zachary's Karate Club graph adjacency matrix.
    """
    G = nx.karate_club_graph()
    return np.asarray(nx.adjacency_matrix(G).todense(), dtype=int)
