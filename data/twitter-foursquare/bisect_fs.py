import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path

    import networkx as nx

    import matplotlib.pyplot as plt

    # -----
    fn_nodes = Path().cwd().resolve() / "nodes_fs.txt"
    fn_edges = Path().cwd().resolve() / "fedges.txt"


@app.cell
def _():
    with open(fn_nodes, "r") as _fh:
        nodes = [
            line.strip()
            for line in _fh.readlines()
        ]
    return (nodes,)


@app.cell
def _(nodes):
    _G = nx.read_edgelist(fn_edges)
    G = nx.subgraph(_G, nodes)
    return (G,)


@app.cell
def _(G):
    print(G)
    return


@app.cell
def _(G):
    _LCC_nodes = sorted(nx.connected_components(G), key=len)[-1]
    LCC = nx.subgraph(G, _LCC_nodes)
    print(LCC)
    return (LCC,)


@app.cell
def _(LCC):
    # A, B = nx.spectral_bisection(LCC)
    A, B = nx.community.spectral_modularity_bipartition(LCC)
    # A, B = nx.community.kernighan_lin_bisection(LCC)
    # A, B = nx.community.asyn_fluidc(LCC, k=2)
    return A, B


@app.cell
def _(A, LCC):
    node_colors = [
        "red"
        if node in A
        else "blue"
        for node in LCC.nodes
    ]
    return (node_colors,)


@app.cell
def _(LCC, node_colors):
    nx.draw(
        LCC,
        node_color=node_colors,
        node_size=5
    )

    plt.gca()
    return


@app.cell
def _(A, B, LCC):
    nx.community.modularity(LCC, [A, B])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
