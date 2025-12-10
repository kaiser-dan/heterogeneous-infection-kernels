"""
Helper functions to populate the contagion database.
"""

import numpy as np
import pandas as pd
import io
from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session

from .ORM import (
    ContagionMetadata,
    OperationOutbreakMetadata,
    TwitterMetadata,
    NodeMapping,
)


# ============================================================================
# Array Compression Utilities
# ============================================================================

def compress_array(arr: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, data=arr)
    buffer.seek(0)
    return buffer.read()


def decompress_array(blob: bytes) -> np.ndarray:
    buffer = io.BytesIO(blob)
    return np.load(buffer, allow_pickle=False)['data']



def _infer_timeseries_freq(index: pd.Index, default: str = "1H") -> str:
    """Infer frequency of a datetime-like index, tolerant of short series."""
    if len(index) < 2:
        return default
    try:
        freq = pd.infer_freq(index)
    except ValueError:
        freq = None
    if freq:
        return freq
    try:
        delta = index[1] - index[0]
        return pd.tseries.frequencies.to_offset(delta).freqstr
    except (TypeError, ValueError):
        return default


def insert_operation_outbreak_run(
    session: Session,
    run_id: str,
    sim_id: int,
    adjacency: pd.DataFrame,
    timeseries: pd.DataFrame,
    weight_type: str = "duration",
    is_directed: bool = False,
    description: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
    exposure_success: Optional[pd.DataFrame] = None,
    exposure_failure: Optional[pd.DataFrame] = None,
) -> OperationOutbreakMetadata:
    """
    Insert an Operation Outbreak simulation run into the database.

    Parameters
    ----------
    session : sqlalchemy.orm.Session
        Database session
    run_id : str
        Unique identifier for this run (e.g., "oo_sim_165")
    sim_id : int
        Original Operation Outbreak simulation ID
    adjacency : pd.DataFrame
        Adjacency matrix from contact_adjacency()
    timeseries : pd.DataFrame
        Time series from infection_time_series()
    weight_type : str
        Weight type used: "duration" or "count"
    is_directed : bool
        Whether the network is directed
    description : str, optional
        Human-readable description
    extra_metadata : dict, optional
        Additional metadata as JSON
    exposure_success : pd.DataFrame, optional
        M matrix of successful exposures.
    exposure_failure : pd.DataFrame, optional
        N matrix of failed exposures.

    Returns
    -------
    OperationOutbreakMetadata
        The created metadata object
    """
    # Infer time series metadata
    timeseries_start = timeseries.index[0]
    timeseries_end = timeseries.index[-1]
    num_timesteps = len(timeseries)

    # Infer frequency from timeseries index
    freq = _infer_timeseries_freq(timeseries.index, default="1H")

    # Create unified node ID mapping
    # Get all nodes from both adjacency and timeseries
    # Note: node IDs are already integers from load_contagion_bulk.py mapping
    adjacency_nodes = set(adjacency.index.unique())
    timeseries_nodes = set(timeseries.columns)
    all_original_node_ids = sorted(adjacency_nodes | timeseries_nodes)
    node_id_map = {orig: idx for idx, orig in enumerate(all_original_node_ids)}

    # Reindex adjacency to use integer node IDs
    adjacency_reindexed = adjacency.copy()
    adjacency_reindexed.index = [node_id_map[nid] for nid in adjacency.index]
    adjacency_reindexed.columns = [node_id_map[nid] for nid in adjacency.columns]

    # Reindex timeseries to use integer node IDs
    # Note: timeseries may have fewer nodes than adjacency (not all nodes get infected)
    timeseries_reindexed = timeseries.copy()
    timeseries_int_node_ids = [node_id_map[nid] for nid in timeseries.columns]
    timeseries_reindexed.columns = timeseries_int_node_ids

    # Compress arrays to binary blobs
    adjacency_blob = compress_array(adjacency_reindexed.values)
    timeseries_blob = compress_array(timeseries_reindexed.values)
    exposure_success_blob = None
    exposure_failure_blob = None
    exposure_shape = None

    if exposure_success is not None and exposure_failure is not None:
        # Reindex to use integer node IDs
        exposure_success_reindexed = exposure_success.reindex(all_original_node_ids).fillna(0)
        exposure_failure_reindexed = exposure_failure.reindex(all_original_node_ids).fillna(0)
        exposure_success_blob = compress_array(exposure_success_reindexed.values)
        exposure_failure_blob = compress_array(exposure_failure_reindexed.values)
        exposure_shape = list(exposure_success_reindexed.shape)

    # Create metadata with binary blobs
    metadata = OperationOutbreakMetadata(
        run_id=run_id,
        dataset_type='operation_outbreak',
        num_nodes=len(all_original_node_ids),
        num_edges=int((adjacency > 0).sum().sum()),
        is_directed=is_directed,
        timeseries_freq=freq,
        timeseries_start=timeseries_start.to_pydatetime(),
        timeseries_end=timeseries_end.to_pydatetime(),
        num_timesteps=num_timesteps,
        description=description,
        extra_metadata=extra_metadata,
        sim_id=sim_id,
        weight_type=weight_type,
        # Binary blob storage (now with integer indices)
        adjacency_blob=adjacency_blob,
        timeseries_blob=timeseries_blob,
        adjacency_shape=list(adjacency_reindexed.shape),
        timeseries_shape=list(timeseries_reindexed.shape),
        timeseries_node_ids=timeseries_int_node_ids,
        exposure_success_blob=exposure_success_blob,
        exposure_failure_blob=exposure_failure_blob,
        exposure_shape=exposure_shape,
    )

    # Create node mappings (from original IDs to integer node IDs)
    node_mappings = []
    for original_id, node_id in node_id_map.items():
        node_mappings.append(NodeMapping(
            run_id=run_id,
            node_id=node_id,
            original_id=str(original_id),
            node_type=None
        ))

    # Add to session
    session.add(metadata)
    session.add_all(node_mappings)

    session.commit()

    return metadata


def insert_twitter_run(
    session: Session,
    run_id: str,
    cascade_id: int,
    adjacency: pd.DataFrame,
    timeseries: pd.DataFrame,
    intervention_hashtag: Optional[str] = None,
    source_bot_id: Optional[str] = None,
    original_tweet_id: Optional[str] = None,
    include_source: bool = True,
    is_directed: bool = True,
    description: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
    bot_ids: Optional[set] = None,
    exposure_success: Optional[pd.DataFrame] = None,
    exposure_failure: Optional[pd.DataFrame] = None,
) -> TwitterMetadata:
    """
    Insert a Twitter cascade run into the database.

    Parameters
    ----------
    session : sqlalchemy.orm.Session
        Database session
    run_id : str
        Unique identifier for this run (e.g., "twitter_cascade_985")
    cascade_id : int
        Original cascade ID from retweets dataset
    adjacency : pd.DataFrame
        Adjacency matrix from follow_adjacency()
    timeseries : pd.DataFrame
        Time series from retweet_time_series()
    intervention_hashtag : str, optional
        Primary hashtag/intervention (e.g., "banksysf")
    source_bot_id : str, optional
        Original Twitter user ID of the bot who posted the tweet
    original_tweet_id : str, optional
        Original tweet ID that started the cascade
    include_source : bool
        Whether source bot was included in timeseries
    is_directed : bool
        Whether the network is directed
    description : str, optional
        Human-readable description
    extra_metadata : dict, optional
        Additional metadata as JSON
    bot_ids : set, optional
        Set of original bot IDs for node type mapping
    exposure_success : pd.DataFrame, optional
        M matrix of successful exposures.
    exposure_failure : pd.DataFrame, optional
        N matrix of failed exposures.

    Returns
    -------
    TwitterMetadata
        The created metadata object
    """
    # Infer time series metadata
    timeseries_start = timeseries.index[0]
    timeseries_end = timeseries.index[-1]
    num_timesteps = len(timeseries)

    # Infer frequency from timeseries index
    freq = _infer_timeseries_freq(timeseries.index, default="1H")

    # Create unified node ID mapping
    # Get all nodes from both adjacency and timeseries
    # Note: node IDs are already integers from load_contagion_bulk.py mapping
    adjacency_nodes = set(adjacency.index.unique())
    timeseries_nodes = set(timeseries.columns)
    all_original_node_ids = sorted(adjacency_nodes | timeseries_nodes)
    node_id_map = {orig: idx for idx, orig in enumerate(all_original_node_ids)}

    # Reindex adjacency to use integer node IDs
    adjacency_reindexed = adjacency.copy()
    adjacency_reindexed.index = [node_id_map[nid] for nid in adjacency.index]
    adjacency_reindexed.columns = [node_id_map[nid] for nid in adjacency.columns]

    # Reindex timeseries to use integer node IDs
    # Note: timeseries may have fewer nodes than adjacency
    timeseries_reindexed = timeseries.copy()
    timeseries_int_node_ids = [node_id_map[nid] for nid in timeseries.columns]
    timeseries_reindexed.columns = timeseries_int_node_ids

    # Compress arrays to binary blobs
    adjacency_blob = compress_array(adjacency_reindexed.values)
    timeseries_blob = compress_array(timeseries_reindexed.values)
    exposure_success_blob = None
    exposure_failure_blob = None
    exposure_shape = None

    if exposure_success is not None and exposure_failure is not None:
        # Reindex to use integer node IDs
        exposure_success_reindexed = exposure_success.reindex(all_original_node_ids).fillna(0)
        exposure_failure_reindexed = exposure_failure.reindex(all_original_node_ids).fillna(0)
        exposure_success_blob = compress_array(exposure_success_reindexed.values)
        exposure_failure_blob = compress_array(exposure_failure_reindexed.values)
        exposure_shape = list(exposure_success_reindexed.shape)

    # Create metadata with binary blobs
    metadata = TwitterMetadata(
        run_id=run_id,
        dataset_type='twitter',
        num_nodes=len(all_original_node_ids),
        num_edges=int((adjacency > 0).sum().sum()),
        is_directed=is_directed,
        timeseries_freq=freq,
        timeseries_start=timeseries_start.to_pydatetime(),
        timeseries_end=timeseries_end.to_pydatetime(),
        num_timesteps=num_timesteps,
        description=description,
        extra_metadata=extra_metadata,
        cascade_id=cascade_id,
        intervention_hashtag=intervention_hashtag,
        source_bot_id=source_bot_id,
        original_tweet_id=original_tweet_id,
        include_source=include_source,
        # Binary blob storage (now with integer indices)
        adjacency_blob=adjacency_blob,
        timeseries_blob=timeseries_blob,
        adjacency_shape=list(adjacency_reindexed.shape),
        timeseries_shape=list(timeseries_reindexed.shape),
        timeseries_node_ids=timeseries_int_node_ids,
        exposure_success_blob=exposure_success_blob,
        exposure_failure_blob=exposure_failure_blob,
        exposure_shape=exposure_shape,
    )

    # Create node mappings (from original IDs to integer node IDs)
    node_mappings = []
    for original_id, node_id in node_id_map.items():
        # Determine node type (bot vs user)
        node_type = None
        if bot_ids:
            node_type = 'bot' if original_id in bot_ids else 'user'

        node_mappings.append(NodeMapping(
            run_id=run_id,
            node_id=node_id,
            original_id=str(original_id),
            node_type=node_type
        ))

    # Add to session
    session.add(metadata)
    session.add_all(node_mappings)

    session.commit()

    return metadata


def get_adjacency_matrix(session: Session, run_id: str) -> pd.DataFrame:
    """
    Retrieve adjacency matrix from database for a given run.

    Parameters
    ----------
    session : sqlalchemy.orm.Session
        Database session
    run_id : str
        Run identifier

    Returns
    -------
    pd.DataFrame
        Adjacency matrix with original node IDs as index/columns
    """
    # Get metadata
    metadata = session.query(ContagionMetadata).filter_by(run_id=run_id).first()
    if not metadata:
        raise ValueError(f"Run {run_id} not found")

    # Decompress adjacency blob
    if metadata.adjacency_blob is None:
        raise ValueError(f"No adjacency data found for run {run_id}")

    adjacency_array = decompress_array(metadata.adjacency_blob)

    # Get node mappings to reconstruct original IDs
    node_mappings = session.query(NodeMapping).filter_by(run_id=run_id).order_by(NodeMapping.node_id).all()

    # Create mapping from integer node_id to original_id
    # Try to convert original_id back to appropriate type
    node_id_to_original = {}
    for nm in node_mappings:
        original_id = nm.original_id
        # Try to convert to int if possible, otherwise keep as string
        try:
            if original_id.isdigit():
                original_id = int(original_id)
            elif original_id.replace('.', '', 1).replace('-', '', 1).isdigit():
                original_id = float(original_id)
        except (ValueError, AttributeError):
            pass
        node_id_to_original[nm.node_id] = original_id

    # Reconstruct DataFrame with original node IDs
    original_node_ids = [node_id_to_original[i] for i in range(len(node_mappings))]

    adjacency = pd.DataFrame(
        adjacency_array,
        index=original_node_ids,
        columns=original_node_ids
    )

    return adjacency


def get_timeseries(session: Session, run_id: str) -> pd.DataFrame:
    """
    Retrieve time series from database for a given run.

    Parameters
    ----------
    session : sqlalchemy.orm.Session
        Database session
    run_id : str
        Run identifier

    Returns
    -------
    pd.DataFrame
        Time series DataFrame (rows=time, cols=nodes, values=0/1)
    """
    # Get metadata
    metadata = session.query(ContagionMetadata).filter_by(run_id=run_id).first()
    if not metadata:
        raise ValueError(f"Run {run_id} not found")

    # Decompress timeseries blob
    if metadata.timeseries_blob is None:
        raise ValueError(f"No timeseries data found for run {run_id}")

    timeseries_array = decompress_array(metadata.timeseries_blob)

    # Create timeline
    timeline = pd.date_range(
        start=metadata.timeseries_start,
        periods=metadata.timeseries_shape[0],
        freq=metadata.timeseries_freq
    )

    # Get node mappings to reconstruct original IDs
    node_mappings = session.query(NodeMapping).filter_by(run_id=run_id).order_by(NodeMapping.node_id).all()

    # Create mapping from integer node_id to original_id
    node_id_to_original = {}
    for nm in node_mappings:
        original_id = nm.original_id
        # Try to convert to int if possible, otherwise keep as string
        try:
            if original_id.isdigit():
                original_id = int(original_id)
            elif original_id.replace('.', '', 1).replace('-', '', 1).isdigit():
                original_id = float(original_id)
        except (ValueError, AttributeError):
            pass
        node_id_to_original[nm.node_id] = original_id

    # Get the integer node IDs present in timeseries
    timeseries_int_node_ids = metadata.timeseries_node_ids

    # Map back to original node IDs
    timeseries_original_ids = [node_id_to_original[int_id] for int_id in timeseries_int_node_ids]

    # Reconstruct DataFrame
    timeseries = pd.DataFrame(
        timeseries_array,
        index=timeline,
        columns=timeseries_original_ids,
        dtype=int
    )

    return timeseries
