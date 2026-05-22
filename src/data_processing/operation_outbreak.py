import pandas as pd
import numpy as np
from pathlib import Path
import networkx as nx


__all__ = [
    "contact_adjacency",
    "infection_time_series",
    "compute_exposure_counts",
    "compute_exposure_counts_from_timeseries",
    "generate_oo_exposure_parquet",
    "contact_edgelist",
    "apply_backbone"
]


def apply_backbone(adj: pd.DataFrame, method: str = "disparity", threshold: float = 0.05) -> pd.DataFrame:
    """
    Apply a network backbone filter.
    Implements a dependency-free backbone filter.
    Default is Disparity Filter (Multiscale Backbone), which preserves heterogeneity.
    """
    if method == "threshold":
        return (adj >= threshold).astype(int)

    # Disparity Filter / Multiscale Backbone
    # Defaulting to this for 'metric' as well to be safe, or if explicitly requested.
    if method in ["disparity", "metric"]:
        # 1. Normalize weights by node strength (degree)
        # p_ij = w_ij / s_i
        strength = adj.sum(axis=1)
        # Avoid division by zero
        p_norm = adj.div(strength, axis=0).fillna(0)
        
        # 2. Calculate alpha_ij for each edge from perspective of node i
        # alpha_ij = (1 - p_ij)^(k_i - 1)
        # where k_i is degree (number of neighbors)
        degree = (adj > 0).sum(axis=1)
        
        # Calculate alpha matrix
        # Create mask of existing edges
        mask = adj > 0
        
        # Helper to compute alpha per row
        def get_alphas(row):
            k = degree[row.name]
            if k <= 1:
                return pd.Series(1.0, index=row.index)
            # Clip probability to 1.0 to avoid complex numbers or warnings
            p = row.clip(upper=1.0)
            return (1 - p)**(k - 1)

        alpha_out = p_norm.apply(get_alphas, axis=1)
        
        # The filter is symmetric in theory but calculation is directed
        # We keep edge if alpha_ij < threshold OR alpha_ji < threshold
        alpha_in = alpha_out.T
        min_alpha = np.minimum(alpha_out, alpha_in)
        
        # Filter: Keep if significance < threshold
        backbone_mask = (mask) & (min_alpha < threshold)
        
        return adj * backbone_mask.astype(int)

    print(f"Warning: Backbone method '{method}' not recognized. Using raw.")
    return adj




def contact_adjacency(histories,sim_id,*,weight="duration",directed=False,user_column="user_id",peer_column="peer_id"):
    """
    Build an adjacency matrix for a single Operation Outbreak simulation.
    ----------
    histories :
        DataFrame containing Operation Outbreak event logs (see ``histories.csv``).
    sim_id :
        Simulation identifier to filter on.
    weight :
        ``"duration"`` sums the ``contact_length`` column (converted to seconds),
        ``"count"`` counts distinct contact events. Defaults to ``"duration"``.
    directed :
        Whether to preserve the inherent direction of the contact logs. Defaults to
        ``False``, which symmetrizes the matrix.
    user_column, peer_column :
        Column names to treat as the ego and alter identifiers.

    Returns
    -------
    pandas.DataFrame
        Square matrix indexed/columned by participant IDs with weights filled in.
    """
    sim_contacts = histories[(histories["sim_id"] == sim_id) & (histories["type"] == "contact")].copy()
    sim_contacts = sim_contacts.dropna(subset=[user_column, peer_column])
    sim_contacts[user_column] = sim_contacts[user_column].astype(str)
    sim_contacts[peer_column] = sim_contacts[peer_column].astype(str)

    if weight == "duration":
        sim_contacts["weight"] = sim_contacts["contact_length"].fillna(0.0) / 1_000.0
    else:
        sim_contacts["weight"] = 1.0
    
    endpoints = pd.concat([sim_contacts[user_column], sim_contacts[peer_column]], ignore_index=True).unique()
    nodes = pd.Index(sorted(endpoints))
    adjacency = pd.DataFrame(data=np.zeros((len(nodes), len(nodes)), dtype=float),index=nodes,columns=nodes)
    
    grouped = (sim_contacts.groupby([user_column, peer_column], as_index=False)["weight"].sum())
    # Iterate through contact events and add them to the adjacency matrix
    for _, row in grouped.iterrows():
        src = row[user_column]
        dst = row[peer_column]
        weight_value = row["weight"]
        adjacency.loc[src, dst] += weight_value
        if not directed and src != dst:
            adjacency.loc[dst, src] += weight_value
    return adjacency


def contact_edgelist(histories, sim_id, *, weight="duration", directed=False, user_column="user_id", peer_column="peer_id"):
    """
    Build a long-form edge list for a single Operation Outbreak simulation.
    """
    sim_contacts = histories[(histories["sim_id"] == sim_id) & (histories["type"] == "contact")].copy()
    sim_contacts = sim_contacts.dropna(subset=[user_column, peer_column])
    sim_contacts[user_column] = sim_contacts[user_column].astype(str)
    sim_contacts[peer_column] = sim_contacts[peer_column].astype(str)

    if weight == "duration":
        sim_contacts["weight"] = sim_contacts["contact_length"].fillna(0.0) / 1_000.0
    else:
        sim_contacts["weight"] = 1.0

    grouped = (sim_contacts.groupby([user_column, peer_column], as_index=False)["weight"].sum())
    grouped["sim_id"] = sim_id
    if not directed:
        # Symmetrize by ensuring small ID is always 'src'
        grouped["u"] = grouped[[user_column, peer_column]].min(axis=1)
        grouped["v"] = grouped[[user_column, peer_column]].max(axis=1)
        grouped = grouped.groupby(["u", "v", "sim_id"], as_index=False)["weight"].sum()
        return grouped.rename(columns={"u": "src", "v": "dst"})
    
    return grouped.rename(columns={user_column: "src", peer_column: "dst"})



def _to_timestamp(value):
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.tz_convert("UTC") if value.tzinfo else value.tz_localize("UTC")
    return pd.to_datetime(value, utc=True)

def infection_time_series(infections,sim_id,*,freq="1D",participants=None,user_column="user_id",time_column="time",start_time=None,end_time=None,):
    """
    Create a discretized infection indicator matrix for a simulation.

    Parameters
    ----------
    infections :
        DataFrame containing infection events (e.g., ``infections_all.csv``).
    sim_id :
        Simulation identifier to filter on.
    freq :
        Pandas offset alias defining the time resolution (``"1H"``, ``"1D"``, etc.).
    participants :
        Optional iterable of user IDs to include even if not infected.
    user_column, time_column :
        Column names holding the participant ID and infection timestamp.
    start_time, end_time :
        Optional overrides for the time axis boundaries. These can be epoch seconds,
        ISO timestamp strings, or ``pandas.Timestamp`` objects.

    Returns
    -------
    pandas.DataFrame
        Rows represent discretized time bins and columns represent participants; values
        are 0 prior to infection and 1 afterwards.
    """
    #Create and format infection data
    sim_infections = infections[infections["sim_id"] == sim_id].copy()
    sim_infections["infection_ts"] = pd.to_datetime(sim_infections[time_column], unit="s", utc=True)
    
    if start_time is None:
        if not sim_infections.empty:
            start_ts = sim_infections["infection_ts"].min()
        else:
            # Fallback to a default or require start_time
            raise ValueError(f"No infection events for sim {sim_id} and no start_time provided.")
    else:
        start_ts = _to_timestamp(start_time)

    if end_time is None:
        if not sim_infections.empty:
            end_ts = sim_infections["infection_ts"].max()
        else:
            raise ValueError(f"No infection events for sim {sim_id} and no end_time provided.")
    else:
        end_ts = _to_timestamp(end_time)
    freq = freq.lower()
    timeline = pd.date_range(start=start_ts, end=end_ts, freq=freq)
    user_ids = set(str(x) for x in sim_infections[user_column].dropna())
    if participants is not None:
        user_ids.update(str(p) for p in participants)
    user_index = sorted(user_ids)
    # Add data to timeseries efficiently
    # Collect infection times to avoid repeated DataFrame slice updates
    # Force user_id to string to match adjacency matrix IDs
    sim_infections[user_column] = sim_infections[user_column].astype(str)
    uid_to_time = sim_infections.set_index(user_column)["infection_ts"].to_dict()
    
    # Pre-allocate as float to allow np.nan if needed, though OO uses 0/1
    # We'll use a numpy array for speed, then convert to DataFrame
    states_arr = np.zeros((len(timeline), len(user_index)), dtype=int)
    
    for j, uid in enumerate(user_index):
        if uid in uid_to_time:
            inf_ts = uid_to_time[uid]
            # Find the first index >= inf_ts
            idx = timeline.searchsorted(inf_ts)
            if idx < len(timeline):
                states_arr[idx:, j] = 1
                
    states = pd.DataFrame(data=states_arr, index=timeline, columns=user_index)
    return states


def compute_exposure_counts_from_timeseries(
    states,
    adjacency,
    kmax,
    mn_mode="survival",
):
    """
    Build M and N matrices from a pre-computed timeseries and adjacency matrix.
    Parameters
    ----------
    states : pandas.DataFrame
        A DataFrame where rows are timesteps and columns are nodes.
    adjacency : pandas.DataFrame
        Square matrix representing node contacts.
    kmax : int
        Exposure cap; counts >= kmax-1 overflow into the last column.
    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        Tuple ``(M, N)`` with index equal to node ids and columns ``0..kmax-1``.
    """
    if kmax < 1:
        raise ValueError("kmax must be >= 1")
    mn_mode = str(mn_mode).lower()
    if mn_mode not in {"survival", "incidence"}:
        raise ValueError(f"mn_mode must be 'survival' or 'incidence'. Got: {mn_mode}")
    
    nodes = list(adjacency.index)
    states = states.reindex(columns=nodes, fill_value=0)
    binary_adj = (adjacency.values > 0).astype(int)

    state_values = states.values.astype(int, copy=False)
    exposure_values = state_values.dot(binary_adj.T)
    adopted_mask = state_values.astype(bool, copy=False)

    V = len(nodes)
    M = np.zeros((V, kmax), dtype=int)
    N = np.zeros((V, kmax), dtype=int)

    for node_idx in range(V):
        exposures = exposure_values[:, node_idx]
        adopted = adopted_mask[:, node_idx].any()

        if mn_mode == "incidence":
            if adopted:
                adopt_bin = adopted_mask[:, node_idx].argmax()
                success_bucket = min(int(exposures[adopt_bin]), kmax - 1)
                M[node_idx, success_bucket] += 1
            else:
                censor_bucket = min(int(exposures[-1]), kmax - 1)
                N[node_idx, censor_bucket] += 1
            continue

        # survival mode (current/default behavior): count every at-risk interval
        if adopted:
            adopt_bin = adopted_mask[:, node_idx].argmax()
            failed = exposures[:adopt_bin]
            if failed.size:
                buckets = np.minimum(failed.astype(int), kmax - 1)
                counts = np.bincount(buckets, minlength=kmax)
                N[node_idx, :kmax] += counts[:kmax]

            success_bucket = min(int(exposures[adopt_bin]), kmax - 1)
            M[node_idx, success_bucket] += 1
        else:
            if exposures.size:
                buckets = np.minimum(exposures.astype(int), kmax - 1)
                counts = np.bincount(buckets, minlength=kmax)
                N[node_idx, :kmax] += counts[:kmax]

    cols = list(range(kmax))
    M_df = pd.DataFrame(M, index=nodes, columns=cols)
    N_df = pd.DataFrame(N, index=nodes, columns=cols)
    return M_df, N_df


def compute_exposure_counts(
    infections,
    sim_id,
    adjacency,
    *,
    freq="1D",
    participants=None,
    user_column="user_id",
    time_column="time",
    start_time=None,
    end_time=None,
    kmax=6,
    mn_mode="survival",
):
    """
    Build M and N matrices counting infections and failures for an OO simulation.
    This is a wrapper function that first builds the time series and then computes
    the exposure counts.

    Parameters
    ----------
    infections : pandas.DataFrame
        Infection events (``infections_all.csv``).
    sim_id : int
        Simulation identifier.
    adjacency : pandas.DataFrame
        Output of ``contact_adjacency`` (square matrix). Non-zero entries indicate
        potential exposures.
    freq, participants, user_column, time_column, start_time, end_time :
        Passed directly to ``infection_time_series``.
    kmax : int
        Exposure cap; counts >= kmax-1 overflow into the last column.

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        Tuple ``(M, N)`` with index equal to node ids and columns ``0..kmax-1``.
    """
    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square")

    adjacency = adjacency.fillna(0)
    adjacency = adjacency.astype(float)

    if not adjacency.index.equals(adjacency.columns):
        adjacency = adjacency.loc[adjacency.index.intersection(adjacency.columns), :]
        adjacency = adjacency.loc[:, adjacency.index]

    nodes = list(adjacency.index)
    if not nodes:
        raise ValueError("adjacency contains no nodes")

    states = infection_time_series(
        infections,
        sim_id,
        freq=freq,
        participants=participants if participants is not None else nodes,
        user_column=user_column,
        time_column=time_column,
        start_time=start_time,
        end_time=end_time,
    )
    
    return compute_exposure_counts_from_timeseries(states, adjacency, kmax, mn_mode=mn_mode)


def generate_oo_exposure_parquet(
    histories_path: Path,
    sim_id: int,
    output_dir: Path,
    freq: str = "30min",
    kmax: int = 10,
    backbone_method: str = None,
    backbone_threshold: float = None,
    mn_mode: str = "survival",
):
    """
    Process O.O. data for a given simulation and save results to a standardized directory.
    Aligns with the Twitter pipeline structure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_events = pd.read_csv(histories_path, low_memory=False)
    
    # Filter for this simulation
    sim_events = all_events[all_events["sim_id"] == sim_id].copy()
    
    # Extract contacts for adjacency
    histories = sim_events[sim_events["type"] == "contact"].copy()
    
    # Extract infections
    infections = sim_events[sim_events["type"] == "infection"].copy()
    infections["user_id"] = infections["user_id"].astype(str)
    
    if infections.empty:
        print(f"Warning: No infections found for sim {sim_id}. Creating empty placeholder files.")
        # Create empty DataFrames for all expected outputs
        pd.DataFrame().to_parquet(output_dir / "exposures.parquet", index=False)
        pd.DataFrame(columns=['src', 'dst', 'sim_id', 'weight']).to_parquet(output_dir / "edges.parquet", index=False)
        pd.DataFrame(columns=['user_id', 'sim_id', 'infection_time', 'adopted']).to_parquet(output_dir / "nodes.parquet", index=False)
        pd.DataFrame().to_parquet(output_dir / "M.parquet", index=True)
        pd.DataFrame().to_parquet(output_dir / "N.parquet", index=True)
        print(f"  Created empty placeholder files for sim {sim_id} in {output_dir}")
        return

    adj = contact_adjacency(histories, sim_id)
    
    # Apply backbone if requested
    if backbone_method:
        print(f"Applying backbone: {backbone_method} (threshold={backbone_threshold})")
        adj = apply_backbone(adj, method=backbone_method, threshold=backbone_threshold)
        
    states = infection_time_series(infections, sim_id, freq=freq, participants=adj.index)
    
    # Ensure states columns exactly match adj index
    states = states.reindex(columns=adj.index, fill_value=0)
    
    nodes = list(adj.index)
    state_values = states.values.astype(int)
    weighted_adj = adj.values # (V, V)
    
    rows = []
    for i, node in enumerate(nodes):
        node_states = state_values[:, i]
        adopted = node_states.any()
        
        if adopted:
            adopt_bin = node_states.argmax()
            peer_states = state_values[adopt_bin, :]
        else:
            peer_states = state_values[-1, :]
            
        incoming_weights = weighted_adj[:, i] # (V,) weights from all j to me i
        exposure_weights = peer_states * incoming_weights
        a_list = exposure_weights[exposure_weights > 0].tolist()
        num_exposures = sum(a_list)
        
        rows.append({
            'user_id': str(node),
            'sim_id': sim_id,
            'adopted': adopted,
            'k': num_exposures, 
            'kappa': len(a_list), 
            'A': a_list 
        })
        
    df = pd.DataFrame(rows)
    df.to_parquet(output_dir / "exposures.parquet", engine='pyarrow', index=False)

    # Export Edges
    edges_rows = []
    for i, src in enumerate(adj.index):
        for j, dst in enumerate(adj.columns):
            if i <= j:  
                weight = adj.iloc[i, j]
                if weight > 0:
                    edges_rows.append({
                        'src': str(src),
                        'dst': str(dst),
                        'sim_id': sim_id,
                        'weight': weight
                    })
    pd.DataFrame(edges_rows).to_parquet(output_dir / "edges.parquet", engine='pyarrow', index=False)

    # Export Nodes (Infection Times)
    inf_map = infections.set_index(infections["user_id"].astype(str))["time"].to_dict()
    node_rows = []
    for node in adj.index:
        node_str = str(node)
        node_rows.append({
            'user_id': node_str,
            'sim_id': sim_id,
            'infection_time': inf_map.get(node_str, np.nan),
            'adopted': node_str in inf_map
        })
    pd.DataFrame(node_rows).to_parquet(output_dir / "nodes.parquet", engine='pyarrow', index=False)

    # 4. Compute and Save M/N Matrices (Fixed q=1.0)
    m_df, n_df = compute_exposure_counts(infections, sim_id, adj, freq=freq, kmax=kmax, mn_mode=mn_mode)
    m_df.to_parquet(output_dir / "M.parquet", engine='pyarrow', index=True)
    n_df.to_parquet(output_dir / "N.parquet", engine='pyarrow', index=True)
    print(f"  Saved simulation results to {output_dir} (mn_mode={mn_mode})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process Operation Outbreak data.")
    parser.add_argument("--histories", type=str, required=True, help="Path to histories.csv")
    parser.add_argument("--sim-id", type=int, required=True, help="Simulation ID")
    parser.add_argument("--output", type=str, required=True, help="Output parquet path")
    parser.add_argument("--freq", type=str, default="30min", help="Time bin frequency (pandas offset alias, e.g. 30min, 1h)")
    parser.add_argument("--backbone-method", type=str, default=None, help="Network backbonining method")
    parser.add_argument("--backbone-threshold", type=float, default=0.05, help="Backbone threshold")
    parser.add_argument("--kmax", type=int, default=10, help="Maximum neighbors for M/N matrices")
    parser.add_argument(
        "--mn-mode",
        type=str,
        default="survival",
        choices=["survival", "incidence"],
        help="How to build M/N matrices.",
    )
    
    args = parser.parse_args()
    
    generate_oo_exposure_parquet(
        histories_path=Path(args.histories),
        sim_id=args.sim_id,
        output_dir=Path(args.output),
        freq=args.freq,
        kmax=args.kmax,
        backbone_method=args.backbone_method,
        backbone_threshold=args.backbone_threshold,
        mn_mode=args.mn_mode,
    )


def generate_stan_matrices_oo(ir_path, config):
    """
    Discretize OO dynamic graph into Stan matrices.
    ir_path is a directory containing edges.parquet and infections.parquet
    """
    ir_path = Path(ir_path)

    # Check for pre-computed M/N matrices
    m_path = ir_path / "M.parquet"
    n_path = ir_path / "N.parquet"

    if m_path.exists() and n_path.exists():
        print(f"Loading pre-computed OO matrices from {ir_path}")
        M_df = pd.read_parquet(m_path)
        N_df = pd.read_parquet(n_path)
        # Convert to numpy and ensure float type for Stan real arrays
        return M_df.values.astype(float), N_df.values.astype(float)

    print(f"Warning: Pre-computed OO matrices not found in {ir_path}. Falling back to dynamic discretization (not yet fully implemented).")
    edges = pd.read_parquet(ir_path / "edges.parquet")
    infections = pd.read_parquet(ir_path / "infections.parquet")

    kmax = config.get('kmax', 20)
    # Mock return for fallback
    V_mock = len(infections['user_id'].unique()) if 'user_id' in infections.columns else 100
    M = np.zeros((V_mock, kmax), dtype=float)
    N = np.zeros((V_mock, kmax), dtype=float)
    return M, N
