
import json
import datetime
from itertools import combinations
import pandas as pd
from scipy.special import binom
import sqlite3
import networkx as nx
import numpy as np
from pathlib import Path

# Constants from original tools.py
interventions = ['getyourflushot', 'highfiveastranger', 'somethinggood',
                 'HowManyPushups', 'turkeyface', 'SFThanks',
                 'blackfridaystories', 'BanksySF']

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
    conn = sqlite3.connect(DATA_DIR / 'targets.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id_str, handle FROM targets')
    id2handle = {int(row[0]): row[1] for row in cursor}
    conn.close()

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
