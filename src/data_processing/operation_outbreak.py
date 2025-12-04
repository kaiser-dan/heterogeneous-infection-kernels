from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "contact_adjacency",
    "infection_time_series",
    "compute_exposure_counts",
]


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
    if weight == "duration":
        sim_contacts["weight"] = sim_contacts["contact_length"].fillna(0.0) / 1_000.0
    else:
        sim_contacts["weight"] = 1.0
    endpoints = pd.concat([sim_contacts[user_column], sim_contacts[peer_column]], ignore_index=True).dropna()
    nodes = pd.Index(sorted(endpoints.unique()))
    adjacency = pd.DataFrame(data=np.zeros((len(nodes), len(nodes)), dtype=float),index=nodes,columns=nodes)
    grouped = (sim_contacts.groupby([user_column, peer_column], as_index=False)["weight"].sum())
    #Iterate through contact events ad add them to the adjacency matrix
    for _, row in grouped.iterrows():
        src = row[user_column]
        dst = row[peer_column]
        weight_value = row["weight"]
        adjacency.loc[src, dst] += weight_value
        if not directed and src != dst:
            adjacency.loc[dst, src] += weight_value
    return adjacency



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
    start_ts = _to_timestamp(start_time) or sim_infections["infection_ts"].min()
    end_ts = _to_timestamp(end_time) or sim_infections["infection_ts"].max()
    freq = freq.lower()
    timeline = pd.date_range(start=start_ts, end=end_ts, freq=freq)
    user_ids = set(sim_infections[user_column].dropna().astype(int).tolist())
    if participants is not None:
        user_ids.update(int(uid) for uid in participants)
    user_index = sorted(user_ids)
    #Add data to timeseries
    states = pd.DataFrame(data=0,index=timeline,columns=user_index,dtype=int)
    for _, event in sim_infections[[user_column, "infection_ts"]].iterrows():
        uid = int(event[user_column])
        infection_ts = event["infection_ts"]
        states.loc[states.index >= infection_ts, uid] = 1
    return states


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
):
    """
    Build M and N matrices counting infections and failures for an OO simulation.

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
    if kmax < 1:
        raise ValueError("kmax must be >= 1")

    adjacency = adjacency.fillna(0)
    adjacency = adjacency.astype(float)

    if not adjacency.index.equals(adjacency.columns):
        adjacency = adjacency.loc[adjacency.index.intersection(adjacency.columns), :]
        adjacency = adjacency.loc[:, adjacency.index]

    nodes = list(adjacency.index)
    if not nodes:
        raise ValueError("adjacency contains no nodes")

    binary_adj = (adjacency.values > 0).astype(int)

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
    states = states.reindex(columns=nodes, fill_value=0)

    state_values = states.values.astype(int, copy=False)
    exposure_values = state_values.dot(binary_adj.T)
    adopted_mask = state_values.astype(bool, copy=False)

    V = len(nodes)
    M = np.zeros((V, kmax), dtype=int)
    N = np.zeros((V, kmax), dtype=int)

    for node_idx in range(V):
        exposures = exposure_values[:, node_idx]
        if adopted_mask[:, node_idx].any():
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


base = Path("data") / "Blanchard-Thesis-main" / "data"
sim_to_demo = 165

# Latest Operation Outbreak export stores both contact histories and cleaned
# infection events in the infections_all.csv file.
histories_df = pd.read_csv(base / "oo-histories-all.csv")
infections_df = pd.read_csv(base / "infections_all.csv")

adjacency = contact_adjacency(
    histories_df,
    sim_to_demo,
    weight="duration",
    directed=False,
)
series = infection_time_series(
    infections_df,
    sim_to_demo,
    freq="1H",
)

print(f"Adjacency for simulation {sim_to_demo}")
print(adjacency.head())
print("\nInfection time series (first rows)")
print(series.head())
import matplotlib.pyplot as plt
plt.spy(series.values)
