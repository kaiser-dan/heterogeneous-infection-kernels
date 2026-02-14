import marimo

__generated_with = "0.19.4"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Twitter Contagion Visualization

    Explore the propagation of hashtags through the Twitter follower graph.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    import sys
    from pathlib import Path
    import io
    return Path, animation, create_engine, mo, nx, pd, plt, sessionmaker, sys


@app.cell
def _(Path, create_engine, sessionmaker, sys):
    # Ensure src is in path
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    # Import project modules
    from src.data_processing.ORM import TwitterMetadata, NodeMapping
    from src.data_processing.database_utils import (
        get_adjacency_matrix,
        get_timeseries,
        decompress_array,
    )

    # Setup DB Connection
    db_path = REPO_ROOT / "data/contagion.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Session = sessionmaker(bind=engine)
    return (
        NodeMapping,
        Session,
        TwitterMetadata,
        decompress_array,
        get_adjacency_matrix,
        get_timeseries,
    )


@app.cell
def _(Session, TwitterMetadata):
    def get_twitter_runs():
        session = Session()
        try:
            # Filter for valid cascades
            runs = (
                session.query(TwitterMetadata)
                .filter(TwitterMetadata.cascade_id != -1)
                .all()
            )

            run_options = {}
            for r in runs:
                label = f"{r.run_id} (#Nodes: {r.num_nodes}, #Steps: {r.num_timesteps})"
                run_options[label] = r.run_id
            return run_options
        finally:
            session.close()

    run_options = get_twitter_runs()
    return (run_options,)


@app.cell
def _(mo, run_options):
    select_run = mo.ui.dropdown(options=run_options, label="Select Cascade")
    select_run
    return (select_run,)


@app.cell
def _(
    NodeMapping,
    Session,
    TwitterMetadata,
    decompress_array,
    get_adjacency_matrix,
    get_timeseries,
    pd,
):
    def load_run_data(run_id):
        session = Session()
        try:
            meta = session.query(TwitterMetadata).filter_by(run_id=run_id).first()
            adj = get_adjacency_matrix(session, run_id)
            ts = get_timeseries(session, run_id)

            M_df = None
            N_df = None
            if meta.exposure_success_blob:
                success_arr = decompress_array(meta.exposure_success_blob)
                M_df = pd.DataFrame(success_arr, index=adj.index)

            if meta.exposure_failure_blob:
                fail_arr = decompress_array(meta.exposure_failure_blob)
                N_df = pd.DataFrame(fail_arr, index=adj.index)

            node_types = {}
            mappings = session.query(NodeMapping).filter_by(run_id=run_id).all()
            for m in mappings:
                node_types[str(m.original_id)] = m.node_type

            return adj, ts, M_df, N_df, node_types, meta.source_bot_id
        finally:
            session.close()
    return (load_run_data,)


@app.cell
def _(animation, mo, nx, plt):
    def plot_cascade_animation(adj, ts, node_types, source_bot_id):
        # Setup Graph
        G = nx.from_pandas_adjacency(adj, create_using=nx.DiGraph)

        # Static Layout
        pos = nx.spring_layout(G, seed=42, k=0.15)

        def get_color(node, infected=False):
            # types keys are strings
            ntype = node_types.get(str(node), "user")
            if str(node) == str(source_bot_id):
                base_color = "red"
            elif ntype == "bot":
                base_color = "orange"
            else:
                base_color = "blue"

            if infected:
                return "lightgreen" if base_color == "blue" else "yellow"
            return base_color

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.axis("off")

        nodes = list(G.nodes())
        # Initial Draw
        node_colors = [get_color(n, infected=False) for n in nodes]
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=30)
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2, arrows=True)
        title = ax.set_title("Time: Start")

        def update(frame_idx):
            current_state = ts.iloc[frame_idx]

            new_colors = []
            for n in nodes:
                is_infected = False
                if n in current_state.index:
                    if current_state[n] > 0:
                        is_infected = True
                new_colors.append(get_color(n, infected=is_infected))

            ax.collections[0].set_facecolor(new_colors)
            title.set_text(f"Time: {ts.index[frame_idx]}")
            return ax.collections[0], title

        # Limit frames
        frames = len(ts)
        if frames > 100:
            step = frames // 50
            frames_indices = range(0, frames, step)
        else:
            frames_indices = range(frames)

        ani = animation.FuncAnimation(
            fig, update, frames=frames_indices, interval=200, blit=False
        )
        plt.close(fig)
        return mo.Html(ani.to_jshtml())
    return (plot_cascade_animation,)


@app.cell
def _(load_run_data, mo, plot_cascade_animation, plt, select_run):
    _output = []
    if select_run.value:
        adj, ts, M, N, node_types, bot_id = load_run_data(select_run.value)

        # Stats
        _output.append(
            mo.md(
                f"""
        **Run**: {select_run.value}
        **Nodes**: {len(adj)}, **Edges**: {adj.sum().sum()}
        **Timesteps**: {len(ts)}
        **Source Bot**: `{bot_id}`
        """
            )
        )

        # 1. Timeseries Plot
        adoption_curve = ts.sum(axis=1)
        fig_ts, ax_ts = plt.subplots(figsize=(8, 3))
        ax_ts.plot(adoption_curve.index, adoption_curve.values, marker="o", markersize=2)
        ax_ts.set_title("Cumulative Adoptions")
        ax_ts.grid(True)

        # 2. Network Animation
        if len(adj) < 800:
            anim_html = plot_cascade_animation(adj, ts, node_types, bot_id)
        else:
            anim_html = mo.md(f"Network too large for browser animation ({len(adj)} nodes).")

        # 3. Matrices
        fig_mat, (ax_m, ax_n) = plt.subplots(1, 2, figsize=(10, 4))

        if M is not None:
            im_m = ax_m.imshow(
                M.values, aspect="auto", interpolation="nearest", cmap="viridis"
            )
            ax_m.set_title("M (Success)")
            plt.colorbar(im_m, ax=ax_m)

        if N is not None:
            im_n = ax_n.imshow(
                N.values, aspect="auto", interpolation="nearest", cmap="magma"
            )
            ax_n.set_title("N (Failures)")
            plt.colorbar(im_n, ax=ax_n)

        _output.append(
            mo.vstack(
                [
                    mo.md("## Adoption Curve"),
                    fig_ts,
                    mo.md("## Propagation Animation"),
                    anim_html,
                    mo.md("## Exposure Matrices"),
                    fig_mat,
                ]
            )
        )
    _output
    return


if __name__ == "__main__":
    app.run()
