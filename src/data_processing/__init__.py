"""
Data processing module for contagion datasets.

This module provides database schema and utilities for storing and
retrieving contagion network data in a unified format.
"""

from .ORM import (
    Base,
    ContagionMetadata,
    OperationOutbreakMetadata,
    TwitterMetadata,
    NodeMapping,
    create_tables,
    drop_tables,
)

from .database_utils import (
    insert_operation_outbreak_run,
    insert_twitter_run,
    get_adjacency_matrix,
    get_timeseries,
    compress_array,
    decompress_array,
)

__all__ = [
    # ORM models
    'Base',
    'ContagionMetadata',
    'OperationOutbreakMetadata',
    'TwitterMetadata',
    'NodeMapping',
    'create_tables',
    'drop_tables',
    # Database utilities
    'insert_operation_outbreak_run',
    'insert_twitter_run',
    'get_adjacency_matrix',
    'get_timeseries',
    'get_exposure_counts',
    'compress_array',
    'decompress_array',
]
