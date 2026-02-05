"""
Twitter Contagion Data Processing - Verified Implementation

This script uses the proven exposure counting logic from replicate_author.py
(which directly replicates the author's find_retweet_prob_from_q_sc.py) and
extends it to compute:
- M/N matrices for k (simple contagion)
- M/N matrices for kappa (complex contagion)
- A-matrices (exposure counts per bot)
- B-vectors (adoption indicators)

All outputs use incidence format (one 1 per user at final exposure level).
"""

import sys
import json
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path

# Configuration
AUTHOR_CODE_DIR = Path("data/twitter_contagion_data/twitter_contagion_code").resolve()
DATA_DIR = Path("data/twitter_contagion_data").resolve()
PROCESSED_DIR = Path("data/processed/twitter").resolve()

# Import Python 3 compatible tools
#TO-DO: make the tools_p3 part of the src code, adding it to src/data_processing/utils.py change these imports 
from . import utils

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def process_intervention(
    intervention,
    G,
    intervention2tweets,
    intervention2user_retweets,
    intervention2bot_retweets,
    bot_IDs,
    active_user_ids,
    q=1.0,
    count_retweets_as_exposures=True
):
    """
    Process a single intervention using verified author logic.
    
    Returns user-level data with:
    - k: number of unique tweet IDs (simple contagion)
    - kappa: number of unique bot sources (complex contagion)
    - bot_exposures: dict mapping bot_id -> count
    - adopted: boolean
    """
    print(f"Processing {intervention}...")
    
    # 1. Track first retweet time per user (adoption detection)
    user2first_retweet = {}
    for rt in intervention2user_retweets[intervention]:
        user = rt['user']['id']
        rt_time = utils.parse_created_at(rt['created_at'])
        if user not in user2first_retweet or rt_time < user2first_retweet[user]:
            user2first_retweet[user] = rt_time
    
    # 2. Track exposures per user
    # user2expose_tweetlist: for k metric (unique tweets)
    # user2expose_botlist: for kappa metric (unique bots)
    # user2bot2tweets: for A-matrix (exposures per bot)
    user2expose_tweetlist = {}
    user2expose_botlist = {}
    user2bot2tweets = {}
    
    # Process original bot tweets
    for tweet in intervention2tweets[intervention]:
        origin_bot = tweet['user']['id']
        origin_tweet = tweet['id']
        
        if origin_bot not in G:
            continue
        
        for user in G.neighbors(origin_bot):
            # NEW: Filter for active users only
            if user not in active_user_ids:
                continue
            
            tweet_time = utils.parse_created_at(tweet['created_at'])
            
            # Skip if exposure happened after adoption
            if user in user2first_retweet and user2first_retweet[user] < tweet_time:
                continue
            
            # Skip non-users (bots, banned accounts)
            if G.nodes[user].get('account_type') != 'user':
                continue
            
            # Track for k metric (unique tweets)
            if user not in user2expose_tweetlist:
                user2expose_tweetlist[user] = []
            user2expose_tweetlist[user].append(origin_tweet)
            
            # Track for kappa metric (unique bots)
            if user not in user2expose_botlist:
                user2expose_botlist[user] = []
            user2expose_botlist[user].append(origin_bot)
            
            # Track for A-matrix (per-bot counts)
            if user not in user2bot2tweets:
                user2bot2tweets[user] = {}
            if origin_bot not in user2bot2tweets[user]:
                user2bot2tweets[user][origin_bot] = []
            user2bot2tweets[user][origin_bot].append(origin_tweet)
    
    # Process bot retweets
    for tweet in intervention2bot_retweets[intervention]:
        proxy_bot = tweet['user']['id']
        origin_bot = tweet['retweeted_status']['user']['id'] if count_retweets_as_exposures else proxy_bot
        original_tweet = tweet['retweeted_status']['id']
        
        if proxy_bot not in G:
            continue
        
        for user in G.neighbors(proxy_bot):
            # NEW: Filter for active users only
            if user not in active_user_ids:
                continue
            
            tweet_time = utils.parse_created_at(tweet['created_at'])
            
            # Skip if exposure happened after adoption
            if user in user2first_retweet and user2first_retweet[user] < tweet_time:
                continue
            
            # Skip non-users
            if G.nodes[user].get('account_type') != 'user':
                continue
            
            # Track for k metric
            if user not in user2expose_tweetlist:
                user2expose_tweetlist[user] = []
            user2expose_tweetlist[user].append(original_tweet)
            
            # Track for kappa metric
            if user not in user2expose_botlist:
                user2expose_botlist[user] = []
            user2expose_botlist[user].append(origin_bot)
            
            # Track for A-matrix
            if user not in user2bot2tweets:
                user2bot2tweets[user] = {}
            if origin_bot not in user2bot2tweets[user]:
                user2bot2tweets[user][origin_bot] = []
            user2bot2tweets[user][origin_bot].append(original_tweet)
    
    # 3. Compute metrics for each user
    user_data = []
    for user in user2expose_tweetlist.keys():
        # k metric (simple contagion): binomial distribution
        N_raw = len(set(user2expose_tweetlist[user]))
        p_dist_k = utils.compute_binomial_distribution(N_raw, q)
        
        # kappa metric (complex contagion): exposure probabilities
        bot_exposures = {
            bot_id: len(set(tweets))
            for bot_id, tweets in user2bot2tweets[user].items()
        }
        A = list(bot_exposures.values())
        p_dist_kappa = utils.compute_exposure_probabilities_fast(A, q)
        
        adopted = user in user2first_retweet
        
        user_data.append({
            'user_id': str(user),
            'p_dist_k': p_dist_k,
            'p_dist_kappa': p_dist_kappa,
            'adopted': adopted,
            'bot_exposures': bot_exposures
        })
    
    return user_data


def save_matrices(user_data, intervention, q=1.0, kmax=50, kappamax=20):
    """
    Save M/N matrices and A/B matrices in probability format.
    """
    if not user_data:
        print(f"No exposure data for {intervention}")
        return
    
    output_dir = PROCESSED_DIR / intervention
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all bot IDs from the data
    all_bot_ids = set()
    for user in user_data:
        all_bot_ids.update(user['bot_exposures'].keys())
    all_bot_ids = sorted(all_bot_ids)
    
    # Build rows for each user
    rows_k = []
    rows_kappa = []
    rows_a = []
    rows_b = []
    
    for user in user_data:
        user_id = user['user_id']
        p_dist_k = user['p_dist_k']
        p_dist_kappa = user['p_dist_kappa']
        adopted = user['adopted']
        
        # M/N for k metric (Probability/Expected Number format)
        row_k = {'user_id': user_id}
        for i in range(1, kmax + 1):
            p_val = p_dist_k.get(i, 0.0)
            if adopted:
                row_k[f'M_{i}'] = float(p_val)
                row_k[f'N_{i}'] = 0.0
            else:
                row_k[f'M_{i}'] = 0.0
                row_k[f'N_{i}'] = float(p_val)
        rows_k.append(row_k)
        
        # M/N for kappa metric (Probability/Expected Number format)
        row_kappa = {'user_id': user_id}
        for i in range(1, kappamax + 1):
            p_val = p_dist_kappa.get(i, 0.0)
            if adopted:
                row_kappa[f'M_{i}'] = float(p_val)
                row_kappa[f'N_{i}'] = 0.0
            else:
                row_kappa[f'M_{i}'] = 0.0
                row_kappa[f'N_{i}'] = float(p_val)
        rows_kappa.append(row_kappa)
        
        # A-matrix (exposure counts per bot - raw counts)
        row_a = {'user_id': user_id}
        for bot_id in all_bot_ids:
            row_a[str(bot_id)] = user['bot_exposures'].get(bot_id, 0)
        rows_a.append(row_a)
        
        # B-vector (adoption indicator)
        rows_b.append({
            'user_id': user_id,
            'adopted': 1 if adopted else 0
        })
    
    # Save k metric M/N
    df_k = pd.DataFrame(rows_k)
    m_cols_k = [c for c in df_k.columns if c.startswith('M_')]
    n_cols_k = [c for c in df_k.columns if c.startswith('N_')]
    
    m_df_k = df_k[['user_id'] + m_cols_k].copy()
    m_df_k.columns = ['user_id'] + [c.replace('M_', '') for c in m_cols_k]
    m_df_k.to_parquet(output_dir / f'M_k_q{q}.parquet')
    
    n_df_k = df_k[['user_id'] + n_cols_k].copy()
    n_df_k.columns = ['user_id'] + [c.replace('N_', '') for c in n_cols_k]
    n_df_k.to_parquet(output_dir / f'N_k_q{q}.parquet')
    
    # Save kappa metric M/N
    df_kappa = pd.DataFrame(rows_kappa)
    m_cols_kappa = [c for c in df_kappa.columns if c.startswith('M_')]
    n_cols_kappa = [c for c in df_kappa.columns if c.startswith('N_')]
    
    m_df_kappa = df_kappa[['user_id'] + m_cols_kappa].copy()
    m_df_kappa.columns = ['user_id'] + [c.replace('M_', '') for c in m_cols_kappa]
    m_df_kappa.to_parquet(output_dir / f'M_kappa_q{q}.parquet')
    
    n_df_kappa = df_kappa[['user_id'] + n_cols_kappa].copy()
    n_df_kappa.columns = ['user_id'] + [c.replace('N_', '') for c in n_cols_kappa]
    n_df_kappa.to_parquet(output_dir / f'N_kappa_q{q}.parquet')
    
    # Save A-matrix and B-vector
    pd.DataFrame(rows_a).to_parquet(output_dir / 'A_matrix.parquet')
    pd.DataFrame(rows_b).to_parquet(output_dir / 'B_vector.parquet')
    
    print(f"Saved {len(user_data)} users for {intervention}")
    print(f"  - M/N_k_q{q}.parquet (kmax={kmax})")
    print(f"  - M/N_kappa_q{q}.parquet (kappamax={kappamax})")
    print(f"  - A_matrix.parquet ({len(all_bot_ids)} bots)")
    print(f"  - B_vector.parquet")


def aggregate_all_interventions(q=1.0, kmax=50, kappamax=20):
    """
    Aggregate M/N matrices across all interventions using DuckDB.

    Saves aggregated results to data/processed/twitter/full/
    treating "full" as if it were another intervention.
    """
    import duckdb

    print(f"\nAggregating interventions with q={q}...")

    con = duckdb.connect()

    # For each metric (k and kappa), aggregate M and N
    for metric, max_level in [('k', kmax), ('kappa', kappamax)]:
        m_glob = str(PROCESSED_DIR / f"*/M_{metric}_q{q}.parquet")
        n_glob = str(PROCESSED_DIR / f"*/N_{metric}_q{q}.parquet")

        # Check if files exist
        try:
            test = con.query(f"SELECT * FROM '{m_glob}' LIMIT 1").df()
        except:
            print(f"  Warning: No data found for metric={metric}, q={q}")
            continue

        # Get column names (exposure levels)
        cols = con.query(f"SELECT * FROM '{m_glob}' LIMIT 0").df().columns
        exposure_cols = [c for c in cols if c not in ['user_id', 'intervention']]

        # Build SUM expressions for aggregation
        sum_expr = ", ".join([f'SUM("{c}") as "{c}"' for c in exposure_cols])

        # Aggregate M: sum across interventions per user
        # IMPORTANT: Exclude 'full' directory to avoid circular reference!
        query_m = f"""
            SELECT user_id, {sum_expr}
            FROM read_parquet('{m_glob}', filename=true)
            WHERE filename NOT LIKE '%/full/%'
            GROUP BY user_id
            ORDER BY user_id
        """
        m_df = con.query(query_m).df()

        # Aggregate N: sum across interventions per user
        # IMPORTANT: Exclude 'full' directory to avoid circular reference!
        query_n = f"""
            SELECT user_id, {sum_expr}
            FROM read_parquet('{n_glob}', filename=true)
            WHERE filename NOT LIKE '%/full/%'
            GROUP BY user_id
            ORDER BY user_id
        """
        n_df = con.query(query_n).df()

        # Save to "full" directory
        output_dir = PROCESSED_DIR / "full"
        output_dir.mkdir(parents=True, exist_ok=True)

        m_df.to_parquet(output_dir / f"M_{metric}_q{q}.parquet")
        n_df.to_parquet(output_dir / f"N_{metric}_q{q}.parquet")

        print(f"  ✓ M/N_{metric}_q{q}.parquet: {len(m_df)} users, {len(exposure_cols)} exposure levels")

    print(f"\nAggregated data saved to: {PROCESSED_DIR / 'full'}")


def build_exposure_matrices(data_dir=None, output_base_dir=None, q=1.0, count_retweets_as_exposures=True):
    """
    Compatibility wrapper to maintain API for existing tests and scripts.
    """
    global DATA_DIR, PROCESSED_DIR
    if data_dir:
        DATA_DIR = Path(data_dir).resolve()
    if output_base_dir:
        PROCESSED_DIR = Path(output_base_dir).resolve()
    
    # CRITICAL: Patch tools_p3 to use the configured data directory
    utils.DATA_DIR = DATA_DIR
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    run_processing(q=q, count_retweets_as_exposures=count_retweets_as_exposures)


def run_processing(q=1.0, count_retweets_as_exposures=True, filter_active_only=True):
    """Main processing function using verified author logic."""
    print("Loading data using author's utils...")
    
    # 1. Load data
    ID2bot_tweet = utils.get_ID2bot_tweet()
    ID2retweets = utils.get_ID2retweets()
    interventions = utils.interventions
    bot_IDs = utils.bot_IDs
    
    # NEW: Determine 'active' users (those who tweeted / retweeted anything in our logs)
    active_user_ids = set()
    if filter_active_only:
        for rts in ID2retweets.values():
            for rt in rts:
                active_user_ids.add(rt['user']['id'])
        print(f"Identified {len(active_user_ids)} active users (with at least 1 retweet).")
    else:
        # If not filtering, we consider every node in the graph as potentially active
        # We'll fill this on demand or use a dummy check
        pass
    
    # 2. Filter tweets by intervention (CASE INSENSITIVE)
    intervention2tweets = {intervention: [] for intervention in interventions}
    
    for k, v in ID2bot_tweet.items():
        hashtags = [d['text'] for d in v['entities']['hashtags']]
        matches = [el for el in interventions if any(el.lower() == h.lower() for h in hashtags)]
        if len(matches) == 0:
            continue
        elif len(matches) >= 2:
            continue  # Skip tweets with multiple intervention hashtags
        intervention2tweets[matches[0]].append(v)
    
    # 3. Categorize retweets
    intervention2user_retweets = {intervention: [] for intervention in interventions}
    intervention2bot_retweets = {intervention: [] for intervention in interventions}
    
    for intervention, tweets in intervention2tweets.items():
        for tweet in tweets:
            _id = tweet['id']
            if _id not in ID2retweets:
                continue
            for rt in ID2retweets[_id]:
                if rt['user']['id'] in bot_IDs:
                    intervention2bot_retweets[intervention].append(rt)
                else:
                    intervention2user_retweets[intervention].append(rt)
    
    # 4. Build graph
    print("Building graph...")
    G = utils.make_graph()
    
    # Ban suspected bot accounts (97.5th percentile of retweet counts)
    all_retweets = []
    for rts in intervention2user_retweets.values():
        all_retweets.extend(rts)
    
    user2rt_counts = Counter([rt['user']['id'] for rt in all_retweets])
    rt_max = np.percentile(list(user2rt_counts.values()), 97.5)
    banned = [k for k, v in user2rt_counts.items() if v > rt_max]
    for n in banned:
        try:
            G.nodes[n]['account_type'] = 'banned'
        except KeyError:
            pass
    
    # 5. Process each intervention
    for intervention in interventions:
        user_data = process_intervention(
            intervention,
            G,
            intervention2tweets,
            intervention2user_retweets,
            intervention2bot_retweets,
            bot_IDs,
            active_user_ids=active_user_ids if filter_active_only else set(G.nodes()),
            q=q,
            count_retweets_as_exposures=count_retweets_as_exposures
        )
        save_matrices(user_data, intervention, q=q)

    # 6. Aggregate all interventions into "full"
    print("\n" + "="*70)
    print("AGGREGATING ALL INTERVENTIONS")
    print("="*70)
    aggregate_all_interventions(q=q)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=float, default=1.0, help="Sensing probability")
    parser.add_argument("--count_retweets", action="store_true", default=True, help="Count retweets as exposures")
    parser.add_argument("--data-dir", type=str, help="Input data directory")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    # Ignored legacy flag for compatibility with Snakemake call if present
    parser.add_argument("--no-bot-retweets", action="store_true", help="Deprecated/Ignored") 
    
    args = parser.parse_args()
    
    # Use the compatibility wrapper to set globals from CLI args
    build_exposure_matrices(
        data_dir=args.data_dir,
        output_base_dir=args.output_dir,
        q=args.q,
        count_retweets_as_exposures=args.count_retweets
    )
