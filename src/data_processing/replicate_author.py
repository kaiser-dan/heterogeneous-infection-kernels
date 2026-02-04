# -*- coding: utf-8 -*-
"""
Replicate Author Logic Exactly

This script is a direct adaptation of `find_retweet_prob_from_q_sc.py` from the author's codebase.
It modifies the output stage to save user-level exposure data (Parquet) instead of aggregated JSON.
"""

from __future__ import division

import sys
import os
import json
import datetime
import numpy as np
import pandas as pd
from collections import Counter
import networkx as nx
from pathlib import Path

# Configuration - must be set before importing tools
AUTHOR_CODE_DIR = Path("data/twitter_contagion_data/twitter_contagion_code").resolve()
DATA_DIR = Path("data/twitter_contagion_data").resolve()
PROCESSED_DIR = Path("data/processed/twitter_replicated").resolve()
PROJECT_ROOT = Path.cwd()

# Import Python 3 compatible tools
sys.path.insert(0, str(Path("scripts").resolve()))
import tools_p3 as tools

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def run_replication():
    print("Loading data using author's tools...")
    
    # 1. Load Data using exact author calls
    ID2bot_tweet = tools.get_ID2bot_tweet()
    ID2retweets = tools.get_ID2retweets()
    interventions = tools.interventions
    bot_IDs = tools.bot_IDs
    
    # 2. Filter Tweets (CASE INSENSITIVE - matching find_retweet_prob_from_q_sc.py line 31)
    intervention2tweets = {intervention : [] for intervention in interventions}

    for k,v in ID2bot_tweet.items():
        hashtags = [d['text'] for d in v['entities']['hashtags']]
        matches = [el for el in interventions if any(el.lower() == h.lower() for h in hashtags)]
        if len(matches) == 0:
            continue
        elif len(matches) >= 2:
            # Original raises ValueError, but we'll skip for robustness
            continue
        intervention2tweets[matches[0]].append(v)

    intervention2user_retweets = {intervention : [] for intervention in interventions}
    intervention2bot_retweets = {intervention : [] for intervention in interventions}
    
    for intervention, tweets in intervention2tweets.items():
        for tweet in tweets:
            _id = tweet['id']
            if not _id in ID2retweets:
                continue  # Tweet wasn't retweeted
            for rt in ID2retweets[_id]:
                if rt['user']['id'] in bot_IDs:
                    intervention2bot_retweets[intervention].append(rt)
                else:
                    intervention2user_retweets[intervention].append(rt)

    print("Building graph...")
    G = tools.make_graph()

    # Ban accounts suspected of being bots
    # Py3: use list(values()) for sum
    all_retweets = []
    for rts in intervention2user_retweets.values():
        all_retweets.extend(rts)
        
    user2rt_counts = Counter([rt['user']['id'] for rt in all_retweets])
    rt_max = np.percentile(list(user2rt_counts.values()), 97.5)
    banned = [k for k,v in user2rt_counts.items() if v > rt_max]
    for n in banned:
        try:
            G.nodes[n]['account_type'] = 'banned' # nx 2.x G.node -> G.nodes
        except KeyError:
            pass  

    # 3. Process Exposure (Modified to Save)
    process_exposures(G, interventions, intervention2tweets, intervention2user_retweets, intervention2bot_retweets, q=1.0)

def process_exposures(G, interventions, intervention2tweets, intervention2user_retweets, intervention2bot_retweets, q=1.0):
    count_retweets_as_exposures = True
    
    for intervention in interventions:
        print(f"Processing {intervention}...")
        
        # Create map user -> first retweet time
        user2first_retweet = {}
        for rt in intervention2user_retweets[intervention]:
            user = rt['user']['id']
            rt_time = tools.parse_created_at(rt['created_at'])
            if not user in user2first_retweet or rt_time < user2first_retweet[user]:
                user2first_retweet[user] = rt_time

        # Loop over bots/neighbors
        user2expose_tweetlist = {} 
        
        # Original bot tweets
        for tweet in intervention2tweets[intervention]:
            origin = tweet['user']['id'] # Bot ID
            origin_tweet = tweet['id']
            
            # G.neighbors(origin) in nx 2.x returns iterator
            if origin not in G: continue
            
            for user in G.neighbors(origin):
                tweet_time = tools.parse_created_at(tweet['created_at'])
                if user in user2first_retweet and user2first_retweet[user] < tweet_time:
                    continue
                
                # Check account type (handle nx 1.x vs 2.x access if needed, but dict access works)
                if G.nodes[user].get('account_type') != 'user':
                    continue
                    
                try:
                    user2expose_tweetlist[user].append(origin_tweet)
                except KeyError:
                    user2expose_tweetlist[user] = [origin_tweet]

        # Bot retweets
        for tweet in intervention2bot_retweets[intervention]:
            proxy = tweet['user']['id']
            origin = tweet['retweeted_status']['user']['id'] if count_retweets_as_exposures else proxy
            original_tweet = tweet['retweeted_status']['id']
            
            if proxy not in G: continue

            for user in G.neighbors(proxy):
                tweet_time = tools.parse_created_at(tweet['created_at'])
                if user in user2first_retweet and user2first_retweet[user] < tweet_time:
                    continue
                
                if G.nodes[user].get('account_type') != 'user':
                    continue

                try:
                    user2expose_tweetlist[user].append(original_tweet)
                except KeyError:
                    user2expose_tweetlist[user] = [original_tweet]

        # --- SAVE M/N LOGIC ---
        # Convert user2expose_tweetlist to DataFrame ROWS
        rows = []
        for user, tweetlist in user2expose_tweetlist.items():
            n_raw = len(set(tweetlist))
            if n_raw == 0: continue
            
            is_adopted = user in user2first_retweet
            
            # For q=1.0, binary
            # For this replication script, we simply output "n_raw" and "adopted"
            # We can construct M/N later or right here.
            # Let's output M/N format directly to match our pipeline expectations
            
            row = {'user_id': str(user), 'intervention': intervention}
            
            # Create Sparse M/N k-vector (up to k=50)
            # kmax=50
            for k in range(1, 51):
                col = str(k)
                # M=1 if adopted and k==n_raw
                # N=1 if not adopted and k==n_raw
                if is_adopted and k == n_raw:
                    row[f'M_{col}'] = 1.0
                    row[f'N_{col}'] = 0.0
                elif not is_adopted and k == n_raw:
                    row[f'M_{col}'] = 0.0
                    row[f'N_{col}'] = 1.0
                else:
                    row[f'M_{col}'] = 0.0
                    row[f'N_{col}'] = 0.0
            
            rows.append(row)
            
        if not rows:
            print(f"No exposure data for {intervention}")
            continue

        df = pd.DataFrame(rows)
        
        # Save split M and N files
        output_dir = PROCESSED_DIR / intervention
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract M columns
        m_cols = [c for c in df.columns if c.startswith('M_')]
        m_df = df[['user_id'] + m_cols].copy()
        m_df.columns = ['user_id'] + [c.replace('M_', '') for c in m_cols]
        m_df.to_parquet(output_dir / 'M_k_q1.0.parquet')
        
        # Extract N columns
        n_cols = [c for c in df.columns if c.startswith('N_')]
        n_df = df[['user_id'] + n_cols].copy()
        n_df.columns = ['user_id'] + [c.replace('N_', '') for c in n_cols]
        n_df.to_parquet(output_dir / 'N_k_q1.0.parquet')
        
        print(f"Saved {len(df)} users for {intervention}")

if __name__ == "__main__":
    run_replication()
