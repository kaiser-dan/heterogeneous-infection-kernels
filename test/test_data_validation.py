import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.data_processing.operation_outbreak import contact_adjacency, infection_time_series

@pytest.fixture
def project_root():
    return Path(__file__).parent.parent

@pytest.fixture
def oo_processed_dir(project_root):
    return project_root / 'data' / 'processed' / 'oo'

@pytest.fixture
def twitter_mn_dir(project_root):
    return project_root / 'data' / 'processed' / 'twitter' / 'full'

# ============================================================================
# Operation Outbreak Validation
# ============================================================================

def test_oo_contact_adjacency_mock():
    """Test contact adjacency matrix construction with mock data"""
    histories = pd.DataFrame([
        {'sim_id': 1, 'type': 'contact', 'user_id': 'A', 'peer_id': 'B', 'contact_length': 100000},
        {'sim_id': 1, 'type': 'contact', 'user_id': 'B', 'peer_id': 'C', 'contact_length': 50000},
    ])
    adj = contact_adjacency(histories, sim_id=1, weight='duration')
    assert 'A' in adj.index and 'B' in adj.index and 'C' in adj.index
    assert adj.loc['A', 'B'] == 100.0
    assert adj.loc['B', 'C'] == 50.0

def test_oo_infection_time_series_mock():
    """Test infection time series construction with mock data"""
    infections = pd.DataFrame([
        {'sim_id': 1, 'user_id': 'A', 'time': 1769179120},
    ])
    states = infection_time_series(infections, sim_id=1, freq='1H', participants=['A', 'B'])
    assert states['A'].sum() > 0
    assert states['B'].sum() == 0

def test_oo_processed_files_exist(oo_processed_dir):
    """Verify that OO simulation data exists"""
    if not oo_processed_dir.exists():
        pytest.skip("OO processed directory not found")

    sim_files = list(oo_processed_dir.glob("sim_165*/exposures.parquet"))
    if not sim_files:
        sim_files = list(oo_processed_dir.glob("sim_165*_exposures.parquet"))
    if not sim_files:
        pytest.skip("No processed OO files found for sim 165")

# ============================================================================
# Twitter Data Validation
# ============================================================================

def test_twitter_mn_structure(twitter_mn_dir):
    """Verify structure of Twitter M/N matrices"""
    if not twitter_mn_dir.exists():
        pytest.skip("Twitter MN directory not found")

    m_files = list(twitter_mn_dir.glob("M_kappa_q*.parquet"))
    if not m_files:
        pytest.skip("Twitter M_kappa parquet not found")

    M_df = pd.read_parquet(m_files[0])
    total_adoptions = M_df.drop(columns='user_id').sum().sum()
    assert 150 < total_adoptions < 400, f"Expected ~211 adoptions, got {total_adoptions}"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
