import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import json
import datetime
from itertools import combinations
from scipy.special import binom
import sqlite3
import networkx as nx

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

# Constants from original tools.py
interventions = ['getyourflushot', 'highfiveastranger', 'somethinggood',
                 'HowManyPushups', 'turkeyface', 'SFThanks',
                 'blackfridaystories', 'BanksySF']

# Default data dir for twitter (relative to project root)
DATA_DIR = Path('data/twitter_contagion_data')

def parse_created_at(created_at):
    return datetime.datetime.strptime(created_at, '%a %b %d %H:%M:%S +0000 %Y')

def parse_timestamp(s):
    try:
        basetime = datetime.datetime.strptime(s, '%Y-%m-%d %H:%M:%S.%f')
    except ValueError:
        basetime = datetime.datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
    return basetime

# Map the ID numer of each bot to their handle (screenname)
bot_ID2handle = {
    2749655899 : 'FitVeganGirl_', 2786086363 : 'AndrenatorC', 2786246510 : 'Van_Jester',
    2786363222 : 'SimonWJorgensen', 2786427865 : 'SuperRexy', 2787456828 : 'matsie_at_dtu',
    2787463746 : 'TheRexyGuy', 2787465462 : 'RealAndersDuck', 2787474126 : 'zakflanigan',
    2787481304 : 'Pralesworth', 2787482192 : 'marcussor', 2787485774 : 'ioapsy',
    2787485978 : 'Awe5omeMike', 2787486036 : 'AnnasHollywood', 2787486060 : 'sonia_manning',
    2787489548 : 'ethanwoods88', 2787498932 : 'LakerolIs', 2787499142 : 'madpopo79',
    2787499932 : 'EmiiiSuu', 2787500612 : 'AxelCyrilian', 2787503576 : 'THINKDEEPYO',
    2787514615 : 'Shtinoehh', 2787516307 : 'SirZenji', 2787527221 : 'RichardsIndie',
    2787531548 : 'CPH_Startup', 2787540506 : 'canuckWong', 2787554990 : 'SusanooGirl',
    2787610574 : 'AlyciaGerald', 2787613616 : 'MyNikkles', 2787620990 : 'meetjamesmet',
    2787739466 : 'tfosterm', 2787833887 : 'ClintCrock', 2787905696 : 'JackBoHorseMan',
    2788164512 : 'hirihiker', 2795790799 : 'hybrishybris', 2801335412 : 'jsmth_t',
    2802511814 : 'neergdave', 2829847496 : 'ericfullhammer', 2841459034 : 'zoesprings'
}
bot_IDs = set(bot_ID2handle.keys())

def get_ID2bot_tweet():
    with open(DATA_DIR / 'all_bot_tweets_', 'r') as f:
        raw = json.load(f)
    return {t['id'] : t for t in raw if not 'retweeted_status' in t}

def get_ID2retweets():
    with open(DATA_DIR / 'retweets_', 'r') as f:
        raw = json.load(f)
    ID2retweets = {}
    for tweetlist in raw:
        if len(tweetlist) == 0: continue
        ID = tweetlist[0]['retweeted_status']['id']
        retweets = sorted(tweetlist, key=lambda t: parse_created_at(t['created_at']))
        ID2retweets[ID] = retweets
    return ID2retweets

def get_interventions():
    return interventions

def get_bot_IDs():
    return bot_IDs

def make_graph(include_details=True):
    G = nx.Graph()
    id2handle = {}
    
    db_path = DATA_DIR / 'targets.db'
    if db_path.exists():
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT id_str, handle FROM targets')
            id2handle = {int(row[0]): row[1] for row in cursor}
            conn.close()
        except sqlite3.Error:
            print(f"Warning: Could not read targets.db at {db_path}. Handles will be missing.")
    else:
        print(f"Note: targets.db not found at {db_path}. Proceeding without handles.")

    edges = pd.read_csv(DATA_DIR / 'edges', low_memory=False)
    for _, row in edges.iterrows():
        try:
            account_id = int(row['account_id'])
            bot_id = int(row['bot_id'])
        except (ValueError, TypeError):
            continue
        
        first_seen = parse_timestamp(row['first_seen'])
        last_seen = parse_timestamp(row['last_seen'])
        relation = row['relation']
        
        if G.has_edge(account_id, bot_id) and include_details:
            link = G[account_id][bot_id]
            first_seen = min(first_seen, link['first_seen'])
            last_seen = max(last_seen, link['last_seen'])
            relation = 'follower' if link['relation'] == 'follower' else 'friend'
        
        if include_details:
            G.add_edge(account_id, bot_id, relation=relation, first_seen=first_seen, last_seen=last_seen)
        else:
            G.add_edge(account_id, bot_id)
            if account_id in id2handle:
                G.nodes[account_id]['handle'] = id2handle[account_id]

    for node in G.nodes():
        G.nodes[node]['account_type'] = 'bot' if node in bot_IDs else 'user'
    return G

def compute_exposure_probabilities_fast(A, q):
    """
    Optimized version of original logic using convolution.
    A is a list of counts of tweets from each unique origin bot.
    Returns {kappa: prob}
    """
    # p_i = probability of seeing at least one tweet from origin i
    probs = [1.0 - (1.0 - q)**a for a in A]
    
    # Convolution to get probability of exactly k successes
    p_dist = np.array([1.0])
    for p in probs:
        p_dist = np.convolve(p_dist, [1.0 - p, p])
    
    return {k: v for k, v in enumerate(p_dist) if v > 0}

def compute_binomial_distribution(N, q):
    return {k: binom(N, k) * (q**k) * ((1.0 - q)**(N - k)) for k in range(N + 1)}
