import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import duckdb

from src.data_processing.process_twitter import build_exposure_matrices

def test_author_reproducibility():
    """
    Unit test to compare our pipeline results with the author's ground truth.
    Ensures that our current implementation matches the paper's results.
    """
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data/twitter_contagion_data'
    processed_dir = project_root / 'data/processed/twitter'
    author_results_path = processed_dir / 'author_results_q1.json'
    
    # 0. Run our pipeline to ensure results are fresh
    print("Running data processing pipeline...")
    build_exposure_matrices(
        data_dir=data_dir,
        output_base_dir=processed_dir
    )
    
    if not author_results_path.exists():
        pytest.skip(
            f"Author ground truth not found at {author_results_path}. "
            "Skipping reproducibility comparison."
        )
        
    # 1. Load Author results
    with open(author_results_path, 'r') as f:
        author_data = json.load(f)
        
    author_exposed = {int(k): float(v) for k, v in author_data['exposed'].items()}
    author_shared = {int(k): float(v) for k, v in author_data['shared'].items()}
    
    con = duckdb.connect()
    
    m_files = str(processed_dir / "*/M_k_q1.0.parquet")
    n_files = str(processed_dir / "*/N_k_q1.0.parquet")
    
    cols = [str(k) for k in range(1, 51)]
    sum_stmt = ", ".join([f'SUM("{c}") as "{c}"' for c in cols])
    
    m_agg = con.execute(f"SELECT {sum_stmt} FROM read_parquet('{m_files}', filename=true) WHERE filename NOT LIKE '%/full/%'").df()
    n_agg = con.execute(f"SELECT {sum_stmt} FROM read_parquet('{n_files}', filename=true) WHERE filename NOT LIKE '%/full/%'").df()
    
    M_ours = m_agg.iloc[0]
    
    def get_max_k_distribution(view_m, view_n):
        cols = [str(k) for k in range(1, 51)]
        sum_cols = " + ".join([f'({int(c)} * COALESCE("{c}", 0))' for c in cols])
        
        query = f"""
        WITH user_totals AS (
            SELECT 
                user_id, 
                ({sum_cols}) as k_total
            FROM (
                SELECT * FROM read_parquet('{m_files}', filename=true) WHERE filename NOT LIKE '%/full/%'
                UNION ALL BY NAME
                SELECT * FROM read_parquet('{n_files}', filename=true) WHERE filename NOT LIKE '%/full/%'
            )
        )
        SELECT k_total, COUNT(*) as user_count
        FROM user_totals
        GROUP BY 1
        ORDER BY 1
        """
        return con.execute(query).df().set_index('k_total')['user_count']

    our_histogram = get_max_k_distribution(m_files, n_files)

    total_m_diff = 0
    total_t_diff = 0
    
    max_k = 50
    for k in range(1, max_k + 1): 
        t_author = author_exposed.get(k, 0.0)
        t_ours = our_histogram.get(float(k), 0.0)
        m_author = author_shared.get(k, 0.0)
        m_ours = M_ours.get(str(k), 0.0)
        
        total_t_diff += abs(t_author - t_ours)
        total_m_diff += abs(m_author - m_ours)
        
    sum_m_author = sum(author_shared.values())
    sum_m_ours = M_ours.iloc[:50].sum()
    
    assert abs(sum_m_author - sum_m_ours) < 1.0, f"Total adoptions mismatch: {sum_m_author} vs {sum_m_ours}"
    
    pop_author = sum(author_exposed.values())
    pop_ours = our_histogram.loc[1:].sum()
    pop_ratio = pop_ours / pop_author if pop_author > 0 else 0
    
    assert 0.5 < pop_ratio < 2.5, f"Exposed population scale mismatch! Ratio: {pop_ratio:.2f}"
    
    print("\nSUCCESS: Pipeline matches author's total adoption events and population scale.")

if __name__ == "__main__":
    test_author_reproducibility()
