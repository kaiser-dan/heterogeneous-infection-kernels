"""
Bulk-load Operation Outbreak simulations and Twitter cascades into the contagion database.

Runs with defaults; no CLI arguments required.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import sys
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_processing import create_tables
from src.data_processing.ORM import ContagionMetadata, TwitterMetadata, NodeMapping
from src.data_processing.database_utils import (
    insert_operation_outbreak_run,
    insert_twitter_run,
    decompress_array,
)
from src.data_processing.operation_outbreak import (
    compute_exposure_counts as compute_exposure_counts_oo,
    contact_adjacency,
    infection_time_series,
)
from src.data_processing.twitter_contagion import (
    DATA_DIR as TW_DATA_DIR,
    build_adjacency_for_participants,
    compute_exposure_counts as compute_exposure_counts_twitter,
    get_bot_ids,
    get_cascade_bot,
    get_cascade_hashtag,
    get_interventions,
    filter_cascades_by_intervention,
    load_bot_tweets,
    load_retweet_cascades,
    retweet_time_series,
)

OO_INCLUDED_SIMS = [129, 131, 136, 139, 140, 141, 143, 148, 149, 155, 158, 159, 162, 163, 165, 46, 49, 50, 53, 54, 55, 56, 57, 58, 59, 61, 64, 65, 66, 67, 68, 69, 72, 74, 75, 79, 81, 82, 83, 85, 88, 118, 126, 127]
# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _run_exists(session, run_id: str) -> bool:
    return session.query(ContagionMetadata).filter_by(run_id=run_id).first() is not None


def _safe_iter(items: Iterable, limit: int | None):
    if limit is None:
        for item in items:
            yield item
    else:
        for idx, item in enumerate(items):
            if idx >= limit:
                break
            yield item


def _create_node_mapping_from_ids(node_ids: set):
    """
    Create a mapping from original node IDs (as strings) to integer node IDs.

    Parameters
    ----------
    node_ids : set
        Set of original node IDs (should be strings)

    Returns
    -------
    tuple[dict, dict]
        (node_to_int, int_to_node) - bidirectional mapping
    """
    sorted_ids = sorted(node_ids)
    node_to_int = {orig_id: idx for idx, orig_id in enumerate(sorted_ids)}
    int_to_node = {idx: orig_id for orig_id, idx in node_to_int.items()}
    return node_to_int, int_to_node


def _create_oo_node_mapping(histories: pd.DataFrame, infections: pd.DataFrame, sim_id: int):
    """
    Create node mapping for an Operation Outbreak simulation.

    Parameters
    ----------
    histories : pd.DataFrame
        Contact history data containing user_id and peer_id columns
    infections : pd.DataFrame
        Infection data containing user_id column
    sim_id : int
        Simulation ID to filter on

    Returns
    -------
    tuple[dict, dict]
        (node_to_int, int_to_node) - bidirectional mapping
    """
    # Get all unique node IDs from contacts
    sim_contacts = histories[(histories["sim_id"] == sim_id) & (histories["type"] == "contact")]
    contact_users = set(str(x) for x in sim_contacts["user_id"].dropna())
    contact_peers = set(str(x) for x in sim_contacts["peer_id"].dropna())

    # Get all unique node IDs from infections
    sim_infections = infections[infections["sim_id"] == sim_id]
    infection_users = set(str(x) for x in sim_infections["user_id"].dropna())

    # Combine all node IDs
    all_node_ids = contact_users | contact_peers | infection_users

    return _create_node_mapping_from_ids(all_node_ids)


def _create_twitter_node_mapping(adjacency: pd.DataFrame, cascade: list):
    """
    Create node mapping for a Twitter cascade.

    Parameters
    ----------
    adjacency : pd.DataFrame
        Adjacency matrix with user IDs as index/columns
    cascade : list
        List of tweet dictionaries in the cascade

    Returns
    -------
    tuple[dict, dict]
        (node_to_int, int_to_node) - bidirectional mapping
    """
    # Get all unique node IDs from adjacency and cascade
    adj_nodes = set(str(x) for x in adjacency.index.unique())

    cascade_users = set()
    for tweet in cascade:
        if "user" in tweet and "id" in tweet["user"]:
            cascade_users.add(str(tweet["user"]["id"]))

    # Combine all node IDs
    all_node_ids = adj_nodes | cascade_users

    return _create_node_mapping_from_ids(all_node_ids)


# ---------------------------------------------------------------------------
# Operation Outbreak bulk load
# ---------------------------------------------------------------------------

def load_all_oo(
    session,
    histories: pd.DataFrame,
    *,
    weight_type: str,
    freq: str,
    directed: bool,
    include_exposures: bool,
    overwrite: bool,
    included_sims: Iterable[int] = OO_INCLUDED_SIMS,
):
    # Extract infection events from histories (type == 'infection')
    infections = histories[histories["type"] == "infection"].copy()

    sim_ids = sorted(set(histories["sim_id"].unique()) & set(infections["sim_id"].unique()))
    print(f"[OO] {len(sim_ids)} simulations with both histories and infections.")

    for sim_id in sim_ids:
        if sim_id not in included_sims:
            continue
        run_id = f"oo_sim_{sim_id}_{weight_type}"
        if not overwrite and _run_exists(session, run_id):
            print(f"[OO][skip] {run_id} already in DB")
            continue

        sim_hist = histories[histories["sim_id"] == sim_id].copy()
        sim_inf = infections[infections["sim_id"] == sim_id].copy()
        if sim_hist.empty or sim_inf.empty:
            print(f"[OO][warn] missing data for sim {sim_id}; histories rows={len(sim_hist)}, infections rows={len(sim_inf)}")
            continue

        # Create node mapping: original_id (str) -> integer node_id
        print(f"[OO][{sim_id}] creating node mapping")
        node_to_int, int_to_node = _create_oo_node_mapping(histories, infections, sim_id)
        print(f"[OO][{sim_id}] mapped {len(node_to_int)} unique nodes")

        # Apply node mapping to convert original IDs to integers
        sim_hist["user_id"] = sim_hist["user_id"].astype(str).map(node_to_int)
        sim_hist["peer_id"] = sim_hist["peer_id"].astype(str).map(node_to_int)
        sim_inf["user_id"] = sim_inf["user_id"].astype(str).map(node_to_int)

        print(f"[OO][{sim_id}] build adjacency (weight={weight_type}, directed={directed})")
        adjacency = contact_adjacency(sim_hist, sim_id, weight=weight_type, directed=directed)
        print(f"[OO][{sim_id}] adjacency {adjacency.shape}, edges {(adjacency > 0).sum().sum()}")

        print(f"[OO][{sim_id}] build infection time series (freq={freq})")
        ts = infection_time_series(sim_inf, sim_id, freq=freq)
        print(f"[OO][{sim_id}] timeseries {ts.shape}")

        exposure_M = exposure_N = None
        if include_exposures:
            print(f"[OO][{sim_id}] build exposure counts (kmax=8)")
            exposure_M, exposure_N = compute_exposure_counts_oo(
                infections=sim_inf,
                sim_id=sim_id,
                adjacency=adjacency,
                freq=freq,
                participants=adjacency.index,
                kmax=8,
            )

        print(f"[OO][{sim_id}] insert run_id={run_id}")
        insert_operation_outbreak_run(
            session,
            run_id=run_id,
            sim_id=sim_id,
            adjacency=adjacency,
            timeseries=ts,
            weight_type=weight_type,
            is_directed=directed,
            description=f"Operation Outbreak sim {sim_id} ({weight_type})",
            exposure_success=exposure_M,
            exposure_failure=exposure_N,
        )
        print(f"[OO][{sim_id}] ok")


# ---------------------------------------------------------------------------
# Twitter bulk load
# ---------------------------------------------------------------------------

from src.data_processing.process_twitter_data import process_intervention

def load_all_twitter_interventions(
    session,
    edges_df: pd.DataFrame,
    retweets: list,
    bot_tweets: list,
    *,
    overwrite: bool,
    limit: int | None,
):
    """
    Load Twitter data aggregated at the intervention level.
    """
    bot_ids = get_bot_ids()
    interventions = get_interventions()
    if limit:
        interventions = interventions[:limit]

    print(f"[TW] Processing {len(interventions)} interventions...")
    for intervention in interventions:
        run_id = f"twitter_{intervention}_aggregated"
        if not overwrite and _run_exists(session, run_id):
            print(f"[TW][skip] {run_id} already in DB")
            continue
        
        process_intervention(session, intervention, retweets, bot_tweets, edges_df, bot_ids)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    db_path = Path("data/contagion.db")
    oo_data_dir = Path("data") / "Blanchard-Thesis-main" / "data"
    tw_data_dir = TW_DATA_DIR

    # Defaults
    oo_weight_type = "duration"
    oo_freq = "1H"
    oo_directed = False

    tw_freq = "6H"
    tw_include_source = True
    tw_limit = 5  # process only 5 cascades for testing
    tw_recency_window = "24H" # Updated default

    include_exposures = True
    overwrite = False

    # Limit OO sims for testing
    included_sims = OO_INCLUDED_SIMS[:2]

    print(f"Using database: {db_path}")
    engine = create_engine(f"sqlite:///{db_path}")
    create_tables(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Operation Outbreak
        histories_path = oo_data_dir / "oo-histories-all.csv"
        if histories_path.exists():
            print(f"\n[OO] Loading data from {oo_data_dir}")
            histories = pd.read_csv(histories_path, low_memory=False)
            load_all_oo(
                session,
                histories,
                weight_type=oo_weight_type,
                freq=oo_freq,
                directed=oo_directed,
                include_exposures=include_exposures,
                overwrite=overwrite,
                included_sims=included_sims,
            )
        else:
            print(f"[OO] Skipping: missing {histories_path}")

        # Twitter
        edges_path = tw_data_dir / "edges"
        retweets_path = tw_data_dir / "retweets_"
        bot_tweets_path = tw_data_dir / "all_bot_tweets_"
        if edges_path.exists() and retweets_path.exists() and bot_tweets_path.exists():
            print(f"\n[TW] Loading data from {tw_data_dir}")
            edges_df = pd.read_csv(edges_path, low_memory=False)
            edges_df["account_id"] = pd.to_numeric(edges_df["account_id"], errors="coerce")
            edges_df["bot_id"] = pd.to_numeric(edges_df["bot_id"], errors="coerce")
            edges_df = edges_df.dropna(subset=["account_id", "bot_id"]).astype({"account_id": "Int64", "bot_id": "Int64"})

            retweets = load_retweet_cascades(filename=retweets_path)
            bot_tweets = load_bot_tweets(filename=bot_tweets_path)

            load_all_twitter_interventions(
                session,
                edges_df,
                retweets,
                bot_tweets,
                overwrite=overwrite,
                limit=tw_limit,
            )
            
        else:
            print(f"[TW] Skipping: missing files in {tw_data_dir}")
    finally:
        session.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
