import networkx as nx
import numpy as np

__all__ = ["sbm", "er", "zkc"]


def sbm(n, k, epsilon, delta=0, seed=None):
    """
    Generates a two-block Stochastic Block Model (SBM) graph.

    Parameters
    ----------
    n : int
        Number of nodes. Split evenly between two blocks.
    k : float
        Average degree of each node.
    epsilon : float
        Controls the ratio of inter- to intra-community edge probability.
        Higher values mean stronger community structure.
    delta : float, optional
        Asymmetry between the two blocks' intra-community probabilities.
        delta=0 (default) gives blocks with equal average degree.
        Positive delta increases p11 (block 1 denser) and decreases p22.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    numpy.ndarray
        Adjacency matrix of the generated SBM graph.
    """
    p = k / (n - 1)
    p12 = (1 - epsilon) * p
    p21 = p12
    p11 = (1 + delta) * (1 + epsilon) * p
    p22 = (1 - delta) * (1 + epsilon) * p

    probs = [[p11, p12], [p21, p22]]
    G = nx.stochastic_block_model([n // 2, n // 2], probs, seed=seed)
    return nx.adjacency_matrix(G).todense()


def er(n, p=None, k=None, seed=None):
    """
    Generates an Erdos-Renyi graph adjacency matrix.

    Parameters
    ----------
    n : int
        Number of nodes.
    p : float, optional
        Edge probability. Provide exactly one of `p` or `k`.
    k : float, optional
        Target average degree. Provide exactly one of `p` or `k`.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    numpy.ndarray
        Adjacency matrix of the generated graph.
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
    Generates Zachary's Karate Club graph adjacency matrix.

    Returns
    -------
    numpy.ndarray
        Adjacency matrix of the Karate Club graph.
    """
    G = nx.karate_club_graph()
    return np.asarray(nx.adjacency_matrix(G).todense(), dtype=int)
