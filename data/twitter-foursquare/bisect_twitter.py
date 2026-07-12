import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import networkx as nx

    import matplotlib.pyplot as plt

    return nx, plt


@app.cell
def _(nx):
    G = nx.read_edgelist("tedges.txt")
    return (G,)


@app.cell
def _(G, nx):
    _LCC_nodes = sorted(nx.connected_components(G), key=len)[-1]
    LCC = nx.subgraph(G, _LCC_nodes)
    return (LCC,)


@app.cell
def _(LCC, nx):
    A, B = nx.community.spectral_modularity_bipartition(LCC)
    # A, B = nx.spectral_bisection(LCC)
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
def _(LCC, node_colors, nx, plt):
    nx.draw(
        LCC,
        node_color=node_colors,
        node_size=5
    )

    plt.gca()
    return


@app.cell
def _(A, B, LCC, nx):
    nx.community.modularity(LCC, [A, B])
    return


if __name__ == "__main__":
    app.run()
