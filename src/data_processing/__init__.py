"""
Data processing module for contagion datasets.

This module provides utilities for processing Twitter and Operation Outbreak 
contagion data into a simplified, interoperable format (Parquet).
"""

from .operation_outbreak import (
    generate_oo_exposure_parquet,
    contact_adjacency,
    infection_time_series
)

__all__ = [
    'generate_oo_exposure_parquet',
    'contact_adjacency',
    'infection_time_series'
]
