"""
SQLAlchemy ORM for contagion datasets.

This module defines the database schema for storing contagion network data
from multiple sources (Operation Outbreak, Twitter, etc.) in a unified format.

The schema consists of four main tables:
1. contagion_metadata: Metadata about each contagion run
2. node_mapping: Maps original node IDs to integer node IDs
3. contagion_timeseries: Time series of infection status for each node

Schema Design Decisions:
- Uses single-table inheritance for metadata (handles different datasets)
- Stores networks as edge lists (not adjacency matrices) for efficiency
- Uses integer timesteps instead of datetimes for timeseries
- Maintains node ID mapping to preserve original identifiers
"""

from sqlalchemy import (
    Column, Integer, String, Boolean, DateTime, Text, ForeignKey,
    Index, JSON, LargeBinary
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class ContagionMetadata(Base):
    """
    Metadata for contagion runs.

    Uses single-table inheritance to handle different dataset types
    (Operation Outbreak, Twitter, etc.) with dataset-specific fields.

    The discriminator column 'dataset_type' determines which subclass
    to use when loading from the database.
    """
    __tablename__ = 'contagion_metadata'

    # Primary key
    run_id = Column(String(255), primary_key=True,
                    comment="Unique identifier for this contagion run")

    # Discriminator for polymorphic identity
    dataset_type = Column(String(50), nullable=False,
                          comment="Type of dataset: 'operation_outbreak', 'twitter', etc.")

    # Common metadata fields (applicable to all datasets)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow,
                        comment="When this record was created")

    num_nodes = Column(Integer, nullable=False,
                       comment="Total number of nodes in the network")

    num_edges = Column(Integer, nullable=False,
                       comment="Total number of edges in the network")

    is_directed = Column(Boolean, nullable=False, default=True,
                         comment="Whether the network is directed")

    # Time series metadata
    timeseries_freq = Column(String(10), nullable=False,
                            comment="Discretization frequency (e.g., '1H', '1D')")

    timeseries_start = Column(DateTime, nullable=False,
                             comment="Start time of the time series (original datetime)")

    timeseries_end = Column(DateTime, nullable=False,
                           comment="End time of the time series (original datetime)")

    num_timesteps = Column(Integer, nullable=False,
                          comment="Total number of timesteps in the time series")

    # Optional fields
    description = Column(Text, nullable=True,
                        comment="Human-readable description of this run")

    # JSON field for additional flexible metadata
    extra_metadata = Column(JSON, nullable=True,
                           comment="Additional dataset-specific metadata as JSON")

    # ===== BINARY BLOB STORAGE FOR MATRICES =====
    adjacency_blob = Column(LargeBinary, nullable=True,
                           comment="Compressed adjacency matrix (numpy.savez_compressed)")

    timeseries_blob = Column(LargeBinary, nullable=True,
                            comment="Compressed time series matrix (numpy.savez_compressed)")

    exposure_success_blob = Column(LargeBinary, nullable=True,
                                   comment="Compressed M matrix (successful infections by exposure count)")

    exposure_failure_blob = Column(LargeBinary, nullable=True,
                                   comment="Compressed N matrix (failed infections by exposure count)")

    # Array metadata for reconstruction
    adjacency_shape = Column(JSON, nullable=True,
                            comment="Shape [n_nodes, n_nodes]")

    timeseries_shape = Column(JSON, nullable=True,
                             comment="Shape [n_timesteps, n_nodes_in_timeseries]")

    timeseries_node_ids = Column(JSON, nullable=True,
                                 comment="List of integer node IDs present in timeseries (subset of all nodes)")

    exposure_shape = Column(JSON, nullable=True,
                            comment="Shape [n_nodes, kmax] for exposure matrices")

    # Relationships
    node_mappings = relationship("NodeMapping", back_populates="contagion_metadata",
                                cascade="all, delete-orphan")
    # Polymorphic configuration
    __mapper_args__ = {
        'polymorphic_identity': 'base',
        'polymorphic_on': dataset_type
    }

    def __repr__(self):
        return f"<ContagionMetadata(run_id='{self.run_id}', type='{self.dataset_type}', nodes={self.num_nodes})>"


class OperationOutbreakMetadata(ContagionMetadata):
    """
    Metadata specific to Operation Outbreak simulations.

    Stores additional fields relevant to O.O. simulations like
    simulation ID, weight type, etc.
    """
    __tablename__ = 'operation_outbreak_metadata'

    run_id = Column(String(255), ForeignKey('contagion_metadata.run_id'), primary_key=True)

    # Operation Outbreak specific fields
    sim_id = Column(Integer, nullable=False,
                   comment="Original simulation ID from Operation Outbreak")

    weight_type = Column(String(20), nullable=True,
                        comment="Weight type used: 'duration' or 'count'")

    __mapper_args__ = {
        'polymorphic_identity': 'operation_outbreak',
    }


class TwitterMetadata(ContagionMetadata):
    """
    Metadata specific to Twitter contagion cascades.

    Stores additional fields relevant to Twitter cascades like
    cascade ID, intervention hashtag, source bot, etc.
    """
    __tablename__ = 'twitter_metadata'

    run_id = Column(String(255), ForeignKey('contagion_metadata.run_id'), primary_key=True)

    # Twitter specific fields
    cascade_id = Column(Integer, nullable=False,
                       comment="Original cascade ID from retweets dataset")

    intervention_hashtag = Column(String(100), nullable=True,
                                 comment="Primary hashtag/intervention (e.g., 'banksysf')")

    source_bot_id = Column(String(50), nullable=True,
                          comment="Original Twitter user ID of the bot who posted the tweet")

    original_tweet_id = Column(String(50), nullable=True,
                              comment="Original tweet ID that started the cascade")

    include_source = Column(Boolean, nullable=False, default=True,
                           comment="Whether source bot was included in timeseries")

    __mapper_args__ = {
        'polymorphic_identity': 'twitter',
    }


class NodeMapping(Base):
    """
    Maps original node identifiers to integer node IDs.

    This table preserves the mapping between original IDs (which might be
    strings, large integers, or UUIDs) and the integer node IDs used in
    the network structure and timeseries tables.

    This allows us to:
    1. Use efficient integer node IDs in the database
    2. Reconstruct the original identifiers when needed
    3. Handle different ID formats across datasets
    """
    __tablename__ = 'node_mapping'

    # Composite primary key
    run_id = Column(String(255), ForeignKey('contagion_metadata.run_id'),
                   primary_key=True, nullable=False)

    node_id = Column(Integer, primary_key=True, nullable=False,
                    comment="Integer node ID (0 to n-1)")

    # Original identifier
    original_id = Column(String(255), nullable=False,
                        comment="Original node identifier from source dataset")

    # Optional node attributes
    node_type = Column(String(50), nullable=True,
                      comment="Type of node (e.g., 'user', 'bot', 'participant')")

    # Relationship
    contagion_metadata = relationship("ContagionMetadata", back_populates="node_mappings")

    # Indexes
    __table_args__ = (
        Index('idx_node_mapping_run_original', 'run_id', 'original_id'),
        Index('idx_node_mapping_run_node', 'run_id', 'node_id'),
    )

    def __repr__(self):
        return f"<NodeMapping(run_id='{self.run_id}', node_id={self.node_id}, original='{self.original_id}')>"


# ============================================================================
# Helper function to create all tables
# ============================================================================

def create_tables(engine):
    """
    Create all tables in the database.

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine
        SQLAlchemy engine connected to the database
    """
    Base.metadata.create_all(engine)


def drop_tables(engine):
    """
    Drop all tables from the database.

    WARNING: This will delete all data!

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine
        SQLAlchemy engine connected to the database
    """
    Base.metadata.drop_all(engine)
