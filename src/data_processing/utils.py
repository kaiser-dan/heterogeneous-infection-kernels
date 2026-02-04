import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

def melt_stan_draws(draws: np.ndarray, param_name: str) -> pd.DataFrame:
    """
    Convert a high-dimensional Stan draws array into a long-form DataFrame.
    
    Shape of draws: (num_draws, dim1, dim2, ...)
    Output columns: ['draw', 'dim1', 'dim2', ..., param_name]
    """
    shape = draws.shape
    num_draws = shape[0]
    dims = shape[1:]
    
    # Create an index for the draws
    draw_idx = np.arange(num_draws)
    
    # Create a multi-index for the dimensions
    # This is a bit tricky for arbitrary dimensions
    # We'll use np.meshgrid or similar to create coordinates
    
    # Simple case: 1D parameter per draw (e.g. pi) -> (draws, K)
    # Result: draw, param_index, value
    
    # General case:
    # Flatten everything but the first dimension
    flattened = draws.reshape(num_draws, -1)
    num_elements = flattened.shape[1]
    
    # Create coordinates for each flattened element
    coords = np.unravel_index(np.arange(num_elements), dims)
    
    rows = []
    # This might be slow for very large arrays, but good for kernels
    # For very large arrays (e.g. latent states), we might want a different approach
    
    # Optimized version using broad-casting/repeat
    df = pd.DataFrame(flattened)
    df.index.name = 'draw'
    df = df.reset_index().melt(id_vars='draw', var_name='flat_index', value_name=param_name)
    
    # Map flat_index back to dimensions
    for i, coord_array in enumerate(coords):
        dim_name = f'dim{i+1}'
        # Map flat_index to the specific coordinate
        mapping = {idx: val for idx, val in enumerate(coord_array)}
        df[dim_name] = df['flat_index'].map(mapping)
    
    df = df.drop(columns=['flat_index'])
    # Reorder columns to put value at the end
    cols = ['draw'] + [f'dim{i+1}' for i in range(len(dims))] + [param_name]
    return df[cols]

def unmelt_stan_draws(df: pd.DataFrame, param_name: str) -> np.ndarray:
    """
    Reverse of melt_stan_draws.
    """
    # Identify dimensions
    dim_cols = [c for c in df.columns if c.startswith('dim')]
    num_draws = df['draw'].nunique()
    
    # Get shape
    shape = [num_draws]
    for col in dim_cols:
        shape.append(df[col].max() + 1)
    
    # Initialize array
    arr = np.zeros(shape)
    
    # This is a bit slow but robust
    # For large arrays, we should use pivot or similar
    
    # Faster way: sort and reshape
    sorted_df = df.sort_values(['draw'] + dim_cols)
    return sorted_df[param_name].values.reshape(shape)

def save_draws_parquet(draws_dict: Dict[str, np.ndarray], output_path: Path):
    """
    Save multiple Stan parameters to one or more Parquet files.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    for name, arr in draws_dict.items():
        df = melt_stan_draws(arr, name)
        # Explicitly cast index columns to int32 to avoid Arrow issues
        for col in df.columns:
            if col == 'draw' or col.startswith('dim'):
                df[col] = df[col].astype(np.int32)
        df.to_parquet(output_path / f"{name}.parquet", index=False, engine='fastparquet')

def load_draws_parquet(input_dir: Path) -> Dict[str, np.ndarray]:
    """
    Load melted draws from a directory into a dictionary of arrays.
    """
    draws = {}
    for p_file in input_dir.glob("*.parquet"):
        name = p_file.stem
        df = pd.read_parquet(p_file, engine='fastparquet')
        draws[name] = unmelt_stan_draws(df, name)
    return draws

def save_node_data_parquet(M: np.ndarray, N: np.ndarray, output_path: Path, node_ids: Optional[List[str]] = None):
    """
    Save M and N matrices as a long-form Parquet file.
    """
    V, kmax = M.shape
    rows = []
    
    # Only save non-zero entries for efficiency if it's sparse? 
    # Actually, M and N are usually small (V x kmax)
    
    data = []
    for i in range(V):
        uid = node_ids[i] if node_ids else str(i)
        for k in range(kmax):
            if M[i, k] > 0 or N[i, k] > 0:
                data.append({
                    'node_id': uid,
                    'k': k,
                    'M': M[i, k],
                    'N': N[i, k]
                })
    
    df = pd.DataFrame(data)
    df.to_parquet(output_path, index=False)

def load_node_data_parquet(path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load M and N matrices from a long-form Parquet file.
    """
    df = pd.read_parquet(path)
    
    node_ids = sorted(df['node_id'].unique())
    id2idx = {uid: i for i, uid in enumerate(node_ids)}
    
    kmax = df['k'].max() + 1
    V = len(node_ids)
    
    M = np.zeros((V, kmax), dtype=int)
    N = np.zeros((V, kmax), dtype=int)
    
    for _, row in df.iterrows():
        idx = id2idx[row['node_id']]
        k = int(row['k'])
        M[idx, k] = int(row['M'])
        N[idx, k] = int(row['N'])
        
    return M, N, node_ids

def get_duckdb_connection(db_path: str = ":memory:"):
    """Get a DuckDB connection and register parquets."""
    import duckdb
    return duckdb.connect(db_path)

def register_oo_views(con, processed_dir: Path):
    """
    Register all OO exposure, edge, and node parquets as views in DuckDB.
    """
    # Exposures
    exp_pattern = str(processed_dir / "oo_sim_*_exposures.parquet")
    con.execute(f"CREATE OR REPLACE VIEW oo_exposures AS SELECT * FROM read_parquet('{exp_pattern}')")
    
    # Edges
    edge_pattern = str(processed_dir / "oo_sim_*_edges.parquet")
    con.execute(f"CREATE OR REPLACE VIEW oo_edges AS SELECT * FROM read_parquet('{edge_pattern}')")
    
    # Nodes
    node_pattern = str(processed_dir / "oo_sim_*_nodes.parquet")
    con.execute(f"CREATE OR REPLACE VIEW oo_nodes AS SELECT * FROM read_parquet('{node_pattern}')")
    
def register_twitter_view(con, processed_dir: Path):
    """
    Register Twitter exposure parquet as a view.
    """
    path = str(processed_dir / "twitter_node_exposures.parquet")
    con.execute(f"CREATE OR REPLACE VIEW twitter_exposures AS SELECT * FROM read_parquet('{path}')")

def register_twitter_matrix_views(con, processed_dir: Path):
    """
    Register the Mk, Nk, Mkappa, Nkappa matrices as views.
    """
    for mat in ['Mk', 'Nk', 'Mkappa', 'Nkappa']:
        path = str(processed_dir / f"twitter/default_data_prep_{mat}.parquet")
        con.execute(f"CREATE OR REPLACE VIEW {mat.lower()} AS SELECT * FROM read_parquet('{path}')")
