import marimo as mo
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
import sys
from pathlib import Path
import io

# Ensure src is in path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Fix imports: use absolute imports from src if needed, but REPO_ROOT in sys.path helps
from src.data_processing.ORM import TwitterMetadata, NodeMapping
from src.data_processing.database_utils import get_adjacency_matrix, get_timeseries, decompress_array

# 1. Setup DB Connection
db_path = REPO_ROOT / "data/contagion.db"
engine = create_engine(f"sqlite:///{db_path}")
Session = sessionmaker(bind=engine)

def get_twitter_runs():
    session = Session()
    try:
        # Filter for runs that are NOT aggregate summaries (cascade_id != -1)
        # Or maybe we want to visualize aggregates too? Aggregates have cascade_id=-1 but M/N matrices.
        # But they lack Adjacency timeseries dynamics usually (we stored dummy ones).
        # Let's show real cascades.
        runs = session.query(TwitterMetadata).filter(TwitterMetadata.cascade_id != -1).all()
        
        run_options = {}
        for r in runs:
            # Create a label with some stats
            label = f"{r.run_id} (#Nodes: {r.num_nodes}, #Steps: {r.num_timesteps})"
            run_options[label] = r.run_id
        return run_options
    finally:
        session.close()

run_options = get_twitter_runs()

# Interactive Selection
select_run = mo.ui.dropdown(options=run_options, label="Select Cascade")

# Load Data Function
def load_run_data(run_id):
    session = Session()
    try:
        meta = session.query(TwitterMetadata).filter_by(run_id=run_id).first()
        adj = get_adjacency_matrix(session, run_id)
        # For timeseries, get_timeseries function handles blob decompression
        ts = get_timeseries(session, run_id)
        
        M_df = None
        N_df = None
        if meta.exposure_success_blob:
             success_arr = decompress_array(meta.exposure_success_blob)
             # The blob stores values. We need indices.
             # In database_utils, we reindexed to original_ids for adjacency.
             # Matrices should match adjacency index.
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

# Visualization
def plot_cascade_animation(adj, ts, node_types, source_bot_id):
    # Setup Graph
    G = nx.from_pandas_adjacency(adj, create_using=nx.DiGraph)
    
    # Position layout (static)
    pos = nx.spring_layout(G, seed=42, k=0.15)
    
    # Color helper
    def get_color(node, infected=False):
        # types keys are strings
        ntype = node_types.get(str(node), 'user')
        if str(node) == str(source_bot_id):
            base_color = 'red' 
        elif ntype == 'bot':
            base_color = 'orange'
        else:
            base_color = 'blue' 
        
        if infected:
            return 'lightgreen' if base_color == 'blue' else 'yellow'
        return base_color

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis('off')
    
    nodes = list(G.nodes())
    # Initial Draw
    node_colors = [get_color(n, infected=False) for n in nodes]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=30)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2, arrows=True)
    title = ax.set_title("Time: Start")

    def update(frame_idx):
        # ts index is datetime, columns are nodes
        current_state = ts.iloc[frame_idx]
        
        new_colors = []
        for n in nodes:
            is_infected = False
            # Check if node exists in time series (it should)
            if n in current_state.index:
                 if current_state[n] > 0:
                     is_infected = True
            new_colors.append(get_color(n, infected=is_infected))
            
        ax.collections[0].set_facecolor(new_colors)
        title.set_text(f"Time: {ts.index[frame_idx]}")
        return ax.collections[0], title

    # Limit frames if too long
    frames = len(ts)
    if frames > 100:
        step = frames // 50
        frames_indices = range(0, frames, step)
    else:
        frames_indices = range(frames)

    ani = animation.FuncAnimation(fig, update, frames=frames_indices, interval=200, blit=False)
    plt.close(fig)
    return mo.Html(ani.to_jshtml())


# Main Layout
mo.md(f"""
# Twitter Contagion Visualization

{select_run}
""")

if select_run.value:
    adj, ts, M, N, node_types, bot_id = load_run_data(select_run.value)
    
    # Stats
    mo.md(f"""
    **Run**: {select_run.value}
    **Nodes**: {len(adj)}, **Edges**: {adj.sum().sum()}
    **Timesteps**: {len(ts)}
    **Source Bot**: `{bot_id}`
    """)
    
    # 1. Timeseries Plot
    adoption_curve = ts.sum(axis=1)
    fig_ts, ax_ts = plt.subplots(figsize=(8, 3))
    ax_ts.plot(adoption_curve.index, adoption_curve.values, marker='o', markersize=2)
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
        im_m = ax_m.imshow(M.values, aspect='auto', interpolation='nearest', cmap='viridis')
        ax_m.set_title("M (Success)")
        plt.colorbar(im_m, ax=ax_m)
        
    if N is not None:
        im_n = ax_n.imshow(N.values, aspect='auto', interpolation='nearest', cmap='magma')
        ax_n.set_title("N (Failures)")
        plt.colorbar(im_n, ax=ax_n)
    
    mo.vstack([
        mo.md("## Adoption Curve"),
        fig_ts,
        mo.md("## Propagation Animation"),
        anim_html,
        mo.md("## Exposure Matrices"),
        fig_mat
    ])
