import networkx as nx
import numpy as np

__all__ = ["sbm", "er", "zkc", "build_graph"]


def sbm(n, k, epsilon, B=2, delta=0, seed=None):
    """
    Generates a B-block Stochastic Block Model (SBM) graph.

    Parameters
    ----------
    n : int
        Number of nodes. Split evenly among B blocks.
    k : float
        Target average degree.
    epsilon : float
        Controls the ratio of inter- to intra-community edge probability.
        Higher values mean stronger community structure.
    B : int
        Number of blocks.
    delta : float, optional
        Asymmetry between block intra-community probabilities. Only
        supported for B=2; ignored when B > 2.
    seed : int, optional

    Returns
    -------
    numpy.ndarray
        (n, n) integer adjacency matrix.
    """
    p     = k / (n - 1)
    p_out = (1.0 - epsilon) * p
    p_in  = (1.0 + n * (B - 1) * epsilon / (n - B)) * p

    if B == 2 and delta != 0:
        p11 = (1 + delta) * p_in
        p22 = (1 - delta) * p_in
        probs = [[p11, p_out], [p_out, p22]]
    else:
        probs = [[p_in if i == j else p_out for j in range(B)] for i in range(B)]

    block_sizes = [n // B] * B
    block_sizes[-1] += n - sum(block_sizes)  # remainder goes to last block
    G = nx.stochastic_block_model(block_sizes, probs, seed=seed)
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


def edgelist(filename):
    """
    Loads graph's adjacency matrix from edgelist file.
    Assumes edgelist file is undirected and unweighted.

    Parameters
    ----------
    filename : str
        Path to edgelist file

    Returns
    -------
    numpy.ndarray
        Adjacency matrix of the loaded graph.
    """
    G = nx.read_edgelist(filename, create_using=nx.Graph)
    return np.asarray(nx.adjacency_matrix(G).todense(), dtype=int)


def build_graph(graph_cfg: dict, seed=None) -> tuple[np.ndarray, int]:
    """Build an adjacency matrix from a config-style graph dict.

    Supported types: 'sbm' (stochastic block model), 'er' (Erdős–Rényi), 'zkc' (Zachary's karate club), or 'edgelist' (loaded from edgelist).

    Example
    -------
        adj, kmax = build_graph({
            "type": "sbm",
            "params": {"n": 500, "avg_degree": 10, "epsilon": 0.05}
        }, seed=42)

    Parameters
    ----------
    graph_cfg : dict with keys 'type' and 'params'.
    seed : int, optional

    Returns
    -------
    adj  : (n, n) integer adjacency matrix.
    kmax : int, maximum degree.
    """
    graph_type = graph_cfg.get("type", "sbm")
    params = graph_cfg.get("params", {})

    if graph_type == "sbm":
        adj = np.asarray(
            sbm(
                n=int(params["n"]),
                k=float(params.get("avg_degree", params.get("k"))),
                epsilon=float(params["epsilon"]),
                B=int(params.get("B", 2)),
                delta=float(params.get("delta", 0.0)),
                seed=seed,
            ),
            dtype=int,
        )

    elif graph_type == "er":
        adj = np.asarray(
            er(
                n=int(params["n"]),
                p=params.get("p", None),
                k=params.get("avg_degree", None),
                seed=seed,
            ),
            dtype=int,
        )

    elif graph_type == "zkc":
        adj = np.asarray(zkc(), dtype=int)

    elif graph_type == "edgelist":
        filename = params.get("filename", "")
        if not filename:
            raise ValueError(
                "If reading in edgelist, must pass 'filename' parameter to edgelist file!"
            )
        adj = np.asarray(edgelist(filename), dtype=int)

    else:
        raise ValueError(f"Unsupported graph type: '{graph_type}'.")

    kmax = int(adj.sum(axis=1).max())
    return adj, kmax

