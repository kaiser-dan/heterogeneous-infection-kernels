#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduce Monsted et al. (2017) Figure 3 observed values from raw data.

Python-3 port of the author's original:
  find_retweet_prob_from_q_sc.py  (SC: k = unique tweet IDs)
  find_retweet_prob_from_q_cc.py  (CC: kappa = unique bot sources)

Design decisions matching the original:
  - NO active_user_ids filter
  - ALL G.neighbors(bot) who are account_type == 'user'
  - SC: N = len(set(tweetlist)); p(k|N,q) = Binomial(N,q)
  - CC: A = Counter(exposers).values(); p(k|A,q) = Poisson-binomial DP
  - Bot retweets counted as exposures (count_retweets_as_exposures=True)
  - Ban filter: top 97.5th percentile of retweet counts
  - q-scan: build exposure maps once, sweep q to find optimal

Outputs:
  data/derived/monsted_fig3_our_computed.csv
  data/derived/monsted_fig3_comparison.csv
"""

from __future__ import division

import json
import datetime
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path
import networkx as nx
from scipy.special import binom as scipy_binom

# ─── Paths ───────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
DATA_DIR    = REPO_ROOT / "data/twitter_contagion_data"
DERIVED_DIR = REPO_ROOT / "data/derived"
DERIVED_DIR.mkdir(parents=True, exist_ok=True)

KMAX_SC = 15
KMAX_CC = 15

# ─── Exposure distribution functions ─────────────────────────────────────────

def compute_binomial_distribution(N, q):
    """Binomial PMF: {k: Pr(k | N, q)} for k = 0..N."""
    result = {}
    for k in range(N + 1):
        p = scipy_binom(N, k) * q**k * (1.0 - q)**(N - k)
        result[k] = p
    return result


def compute_exposure_probabilities(A_tuple, q):
    """Poisson-binomial DP for CC.

    A_tuple = sorted tuple of tweet counts per unique bot.
    Returns {k: Pr(user sees exactly k bots)} for k = 0..|A|.
    P(sees bot i) = 1 - (1-q)^{a_i}.
    """
    p_fail = 1.0 - q
    success_probs = [1.0 - p_fail**a for a in A_tuple]
    n = len(success_probs)
    dp = [0.0] * (n + 1)
    dp[0] = 1.0
    for p_i in success_probs:
        new_dp = [0.0] * (n + 1)
        for j in range(n + 1):
            if dp[j] == 0.0:
                continue
            new_dp[j] += dp[j] * (1.0 - p_i)
            if j < n:
                new_dp[j + 1] += dp[j] * p_i
        dp = new_dp
    return {k: dp[k] for k in range(n + 1)}


# ─── Data / graph helpers ─────────────────────────────────────────────────────

interventions = ['getyourflushot', 'highfiveastranger', 'somethinggood',
                 'HowManyPushups', 'turkeyface', 'SFThanks',
                 'blackfridaystories', 'BanksySF']

bot_ID2handle = {
    2749655899:'FitVeganGirl_', 2786086363:'AndrenatorC', 2786246510:'Van_Jester',
    2786363222:'SimonWJorgensen', 2786427865:'SuperRexy', 2787456828:'matsie_at_dtu',
    2787463746:'TheRexyGuy', 2787465462:'RealAndersDuck', 2787474126:'zakflanigan',
    2787481304:'Pralesworth', 2787482192:'marcussor', 2787485774:'ioapsy',
    2787485978:'Awe5omeMike', 2787486036:'AnnasHollywood', 2787486060:'sonia_manning',
    2787489548:'ethanwoods88', 2787498932:'LakerolIs', 2787499142:'madpopo79',
    2787499932:'EmiiiSuu', 2787500612:'AxelCyrilian', 2787503576:'THINKDEEPYO',
    2787514615:'Shtinoehh', 2787516307:'SirZenji', 2787527221:'RichardsIndie',
    2787531548:'CPH_Startup', 2787540506:'canuckWong', 2787554990:'SusanooGirl',
    2787610574:'AlyciaGerald', 2787613616:'MyNikkles', 2787620990:'meetjamesmet',
    2787739466:'tfosterm', 2787833887:'ClintCrock', 2787905696:'JackBoHorseMan',
    2788164512:'hirihiker', 2795790799:'hybrishybris', 2801335412:'jsmth_t',
    2802511814:'neergdave', 2829847496:'ericfullhammer', 2841459034:'zoesprings',
}
bot_IDs = set(bot_ID2handle.keys())


def parse_created_at(s):
    return datetime.datetime.strptime(s, '%a %b %d %H:%M:%S +0000 %Y')


def load_bot_tweets():
    with open(DATA_DIR / 'all_bot_tweets_', 'r') as f:
        raw = json.load(f)
    return {t['id']: t for t in raw if 'retweeted_status' not in t}


def load_retweets():
    path = DATA_DIR / 'retweets.json'
    if not path.exists():
        path = DATA_DIR / 'retweets_'
    with open(path, 'r') as f:
        raw = json.load(f)
    ID2retweets = {}
    for tweetlist in raw:
        if not tweetlist:
            continue
        tweet_id = tweetlist[0]['retweeted_status']['id']
        ID2retweets[tweet_id] = sorted(tweetlist,
                                       key=lambda t: parse_created_at(t['created_at']))
    return ID2retweets


def make_graph():
    """Build undirected bot-user graph (include_details=False: ignore timestamps)."""
    G = nx.Graph()
    edges = pd.read_csv(DATA_DIR / 'edges', low_memory=False)
    for _, row in edges.iterrows():
        try:
            G.add_edge(int(row['account_id']), int(row['bot_id']))
        except (ValueError, TypeError):
            continue
    for node in G.nodes():
        G.nodes[node]['account_type'] = 'bot' if node in bot_ID2handle else 'user'
    return G


# ─── Phase 1: build q-independent exposure maps ───────────────────────────────

def build_exposure_maps(count_retweets_as_exposures=True):
    """Graph traversal (slow, done once).

    Returns a list of per-intervention records:
      {
        'user2N_sc':   {user_id: int}         # unique tweet IDs seen
        'user2A_cc':   {user_id: tuple}        # sorted counts per unique bot
        'adopted':     set of user_ids who retweeted
      }
    """
    print("Loading data...")
    ID2bot_tweet = load_bot_tweets()
    ID2retweets  = load_retweets()

    # Sort bot tweets by intervention
    intervention2tweets = {iv: [] for iv in interventions}
    for tweet in ID2bot_tweet.values():
        hashtags = [d['text'] for d in tweet['entities']['hashtags']]
        matches  = [iv for iv in interventions
                    if any(iv.lower() == h.lower() for h in hashtags)]
        if len(matches) == 1:
            intervention2tweets[matches[0]].append(tweet)

    # Categorise retweets
    intervention2user_rts = {iv: [] for iv in interventions}
    intervention2bot_rts  = {iv: [] for iv in interventions}
    for iv, tweets in intervention2tweets.items():
        for tweet in tweets:
            if tweet['id'] not in ID2retweets:
                continue
            for rt in ID2retweets[tweet['id']]:
                if rt['user']['id'] in bot_IDs:
                    intervention2bot_rts[iv].append(rt)
                else:
                    intervention2user_rts[iv].append(rt)

    # Build graph and ban top retweeters
    print("Building graph...")
    G = make_graph()
    all_user_rts = [rt for rts in intervention2user_rts.values() for rt in rts]
    user2rt_counts = Counter(rt['user']['id'] for rt in all_user_rts)
    rt_max = np.percentile(list(user2rt_counts.values()), 97.5)
    for uid, cnt in user2rt_counts.items():
        if cnt > rt_max and uid in G.nodes:
            G.nodes[uid]['account_type'] = 'banned'

    maps = []
    for iv in interventions:
        print(f"  Building map for {iv}...")

        # First retweet time per user
        user2first_rt = {}
        for rt in intervention2user_rts[iv]:
            uid = rt['user']['id']
            t   = parse_created_at(rt['created_at'])
            if uid not in user2first_rt or t < user2first_rt[uid]:
                user2first_rt[uid] = t

        # Per-user exposure lists (raw)
        user2tweetlist = {}  # SC
        user2exposers  = {}  # CC

        for tweet in intervention2tweets[iv]:
            origin_bot   = tweet['user']['id']
            origin_tweet = tweet['id']
            tweet_time   = parse_created_at(tweet['created_at'])
            if origin_bot not in G:
                continue
            for user in G.neighbors(origin_bot):
                if G.nodes[user].get('account_type') != 'user':
                    continue
                if user in user2first_rt and user2first_rt[user] < tweet_time:
                    continue
                user2tweetlist.setdefault(user, []).append(origin_tweet)
                user2exposers.setdefault(user, []).append(origin_bot)

        for tweet in intervention2bot_rts[iv]:
            proxy_bot  = tweet['user']['id']
            origin_bot = (tweet['retweeted_status']['user']['id']
                          if count_retweets_as_exposures else proxy_bot)
            orig_tweet = tweet['retweeted_status']['id']
            tweet_time = parse_created_at(tweet['created_at'])
            if proxy_bot not in G:
                continue
            for user in G.neighbors(proxy_bot):
                if G.nodes[user].get('account_type') != 'user':
                    continue
                if user in user2first_rt and user2first_rt[user] < tweet_time:
                    continue
                user2tweetlist.setdefault(user, []).append(orig_tweet)
                user2exposers.setdefault(user, []).append(origin_bot)

        # Compress to q-independent summaries
        user2N_sc = {u: len(set(tl)) for u, tl in user2tweetlist.items()}
        user2A_cc = {u: tuple(sorted(Counter(ex).values()))
                     for u, ex in user2exposers.items()}

        maps.append({
            'user2N_sc': user2N_sc,
            'user2A_cc': user2A_cc,
            'adopted':   set(user2first_rt.keys()),
        })

    return maps


# ─── Phase 2: apply q weighting (fast, called for each q) ────────────────────

def apply_q_weighting(maps, q_sc, q_cc=None):
    """Apply binomial/Poisson-binomial weighting.

    q_sc: q for Simple Contagion (Binomial)
    q_cc: q for Complex Contagion (Poisson-binomial); defaults to q_sc if None.
    Caches distributions by unique N and unique A across all interventions.
    Returns sc_result, cc_result each shaped {'exposed': {k: float}, 'shared': {k: float}}.
    """
    if q_cc is None:
        q_cc = q_sc

    # Collect unique N and A values for caching
    unique_N = set()
    unique_A = set()
    for iv in maps:
        unique_N.update(iv['user2N_sc'].values())
        unique_A.update(iv['user2A_cc'].values())

    # Precompute distributions for all unique values at their respective q
    N_dist = {N: compute_binomial_distribution(N, q_sc) for N in unique_N}
    A_dist = {A: compute_exposure_probabilities(A, q_cc) for A in unique_A}

    sc_exposed = {}
    sc_shared  = {}
    cc_exposed = {}
    cc_shared  = {}

    for iv in maps:
        adopted = iv['adopted']

        for user, N in iv['user2N_sc'].items():
            is_adopted = user in adopted
            for k, p_k in N_dist[N].items():
                if k == 0:
                    continue
                sc_exposed[k] = sc_exposed.get(k, 0.0) + p_k
                if is_adopted:
                    sc_shared[k] = sc_shared.get(k, 0.0) + p_k

        for user, A in iv['user2A_cc'].items():
            is_adopted = user in adopted
            for k, p_k in A_dist[A].items():
                if k == 0:
                    continue
                cc_exposed[k] = cc_exposed.get(k, 0.0) + p_k
                if is_adopted:
                    cc_shared[k] = cc_shared.get(k, 0.0) + p_k

    return ({'exposed': sc_exposed, 'shared': sc_shared},
            {'exposed': cc_exposed, 'shared': cc_shared})


# ─── q-scan: find optimal q vs transcription ─────────────────────────────────

def compute_sse(sc, cc, target):
    """Sum of squared errors vs transcribed figure values.

    Panel A: compare 100*fraction vs transcribed %
    Panel B: compare N_adopted vs transcribed count
    """
    sse = 0.0
    for row in target.itertuples(index=False):
        panel, model, k, observed = row.panel, row.model, int(row.x), float(row.observed)
        data = sc if model == 'SC' else cc
        T = data['exposed'].get(k, 0.0)
        M = data['shared'].get(k, 0.0)
        if T == 0:
            continue
        if panel == 'A':
            sse += (100.0 * M / T - observed) ** 2
        else:
            sse += (M - observed) ** 2
    return sse


def compute_panel_b_max_abs_error(sc, cc, target, model=None):
    """Max absolute error for Panel B only.

    model: 'SC', 'CC', or None (both).
    """
    max_err = 0.0
    for row in target.itertuples(index=False):
        if row.panel != 'B':
            continue
        if model and row.model != model:
            continue
        k  = int(row.x)
        ob = float(row.observed)
        data = sc if row.model == 'SC' else cc
        M = data['shared'].get(k, 0.0)
        max_err = max(max_err, abs(M - ob))
    return max_err


def find_optimal_q(maps, target, q_values):
    """Scan q values and return the one minimising SSE vs transcription."""
    print(f"\nScanning {len(q_values)} q values...")
    best_q   = None
    best_sse = float('inf')
    for q in q_values:
        sc, cc = apply_q_weighting(maps, q)
        sse = compute_sse(sc, cc, target)
        print(f"  q={q:.3f}  SSE={sse:.4f}")
        if sse < best_sse:
            best_sse = sse
            best_q   = q
    print(f"\nOptimal q = {best_q:.3f}  (SSE = {best_sse:.4f})")
    return best_q


def find_optimal_q_panel_b(maps, target, q_values, model):
    """Scan q values minimising Panel B max absolute error for one model (SC or CC)."""
    print(f"\nPanel B scan for {model} over {len(q_values)} q values...")
    best_q   = None
    best_err = float('inf')
    for q in q_values:
        sc, cc = apply_q_weighting(maps, q)
        err = compute_panel_b_max_abs_error(sc, cc, target, model=model)
        print(f"  q={q:.4f}  Panel-B-{model}-maxerr={err:.4f}")
        if err < best_err:
            best_err = err
            best_q   = q
    print(f"\nBest q_{model} = {best_q:.4f}  (Panel B max abs err = {best_err:.4f})")
    return best_q


# ─── Reporting ───────────────────────────────────────────────────────────────

def format_results(sc, cc, q, kmax_sc=KMAX_SC, kmax_cc=KMAX_CC):
    print("\n" + "="*72)
    print(f"SIMPLE CONTAGION (k = unique tweet IDs)  [q = {q:.3f}]")
    print("="*72)
    print(f"{'k':>4}  {'N_exposed':>12}  {'N_adopted':>10}  {'pct_rt (%)':>12}")
    rows_sc = []
    for k in range(1, kmax_sc + 1):
        T = sc['exposed'].get(k, 0.0)
        M = sc['shared'].get(k, 0.0)
        p = M / T if T > 0 else float('nan')
        print(f"{k:>4}  {T:>12.1f}  {M:>10.3f}  {100*p:>12.4f}")
        rows_sc.append({'model': 'SC', 'k': k, 'N_exposed': T, 'N_adopted': M,
                        'frac_retweeted': p, 'pct_retweeted': 100 * p})

    print("\n" + "="*72)
    print(f"COMPLEX CONTAGION (kappa = unique bots)  [q = {q:.3f}]")
    print("="*72)
    print(f"{'k':>4}  {'N_exposed':>12}  {'N_adopted':>10}  {'pct_rt (%)':>12}")
    rows_cc = []
    for k in range(1, kmax_cc + 1):
        T = cc['exposed'].get(k, 0.0)
        M = cc['shared'].get(k, 0.0)
        p = M / T if T > 0 else float('nan')
        print(f"{k:>4}  {T:>12.1f}  {M:>10.3f}  {100*p:>12.4f}")
        rows_cc.append({'model': 'CC', 'k': k, 'N_exposed': T, 'N_adopted': M,
                        'frac_retweeted': p, 'pct_retweeted': 100 * p})

    return rows_sc + rows_cc


def compare_with_transcription(our_rows, transcription_path):
    target = pd.read_csv(transcription_path, comment='#')

    print("\n" + "="*80)
    print("MISMATCH TABLE: our computed vs transcribed figure values")
    print("="*80)

    rows = []
    for trow in target.itertuples(index=False):
        panel, model, x = trow.panel, trow.model, int(trow.x)
        obs_target = float(trow.observed)

        our_row = next((r for r in our_rows if r['model'] == model and r['k'] == x), None)
        if our_row is None:
            print(f"  [{panel},{model},k={x}] NOT FOUND")
            continue

        if panel == 'A':
            our_val = our_row['pct_retweeted']   # already in %
        else:
            our_val = our_row['N_adopted']

        delta   = our_val - obs_target
        pct_err = 100.0 * delta / obs_target if obs_target else float('nan')
        rows.append({'panel': panel, 'model': model, 'k': x,
                     'target': obs_target, 'ours': round(our_val, 5),
                     'delta': round(delta, 5), 'pct_err': round(pct_err, 1)})

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    transcription_path = DERIVED_DIR / 'monsted_fig3_transcribed.csv'
    target = pd.read_csv(transcription_path, comment='#')

    # Phase 1: build exposure maps (graph traversal, q-independent)
    maps = build_exposure_maps()

    # Phase 2a: joint coarse+fine SSE scan
    q_coarse = np.round(np.arange(0.10, 0.51, 0.01), 3)
    best_q_coarse = find_optimal_q(maps, target, q_coarse)
    q_fine = np.round(np.arange(best_q_coarse - 0.01,
                                best_q_coarse + 0.011, 0.002), 4)
    best_q_joint = find_optimal_q(maps, target, q_fine)

    # Phase 2b: independent Panel B optimisation for SC and CC
    q_scan = np.round(np.arange(0.05, 0.56, 0.01), 3)
    best_q_sc = find_optimal_q_panel_b(maps, target, q_scan, model='SC')
    best_q_cc = find_optimal_q_panel_b(maps, target, q_scan, model='CC')

    # Fine-tune each
    for best, model in [(best_q_sc, 'SC'), (best_q_cc, 'CC')]:
        q_fine2 = np.round(np.arange(best - 0.02, best + 0.021, 0.002), 4)
        refined = find_optimal_q_panel_b(maps, target, q_fine2, model=model)
        if model == 'SC':
            best_q_sc = refined
        else:
            best_q_cc = refined

    print(f"\n=== Strategy comparison ===")
    for label, q_sc, q_cc in [
        ('joint',  best_q_joint, best_q_joint),
        ('split',  best_q_sc,    best_q_cc),
    ]:
        sc, cc = apply_q_weighting(maps, q_sc, q_cc)
        maxerr = compute_panel_b_max_abs_error(sc, cc, target)
        sc_err = compute_panel_b_max_abs_error(sc, cc, target, model='SC')
        cc_err = compute_panel_b_max_abs_error(sc, cc, target, model='CC')
        print(f"  {label}: q_SC={q_sc:.4f}  q_CC={q_cc:.4f}  "
              f"Panel-B maxerr={maxerr:.3f} (SC={sc_err:.3f}, CC={cc_err:.3f})")

    # Phase 3: use split q if it improves Panel B, else joint
    sc_joint, cc_joint = apply_q_weighting(maps, best_q_joint)
    sc_split, cc_split = apply_q_weighting(maps, best_q_sc, best_q_cc)
    err_joint = compute_panel_b_max_abs_error(sc_joint, cc_joint, target)
    err_split = compute_panel_b_max_abs_error(sc_split, cc_split, target)

    if err_split < err_joint:
        sc, cc = sc_split, cc_split
        best_q_label = f"SC={best_q_sc:.4f} CC={best_q_cc:.4f} (split)"
        print(f"\nUsing split q: {best_q_label}")
    else:
        sc, cc = sc_joint, cc_joint
        best_q_label = f"{best_q_joint:.4f} (joint)"
        print(f"\nUsing joint q: {best_q_label}")

    our_rows = format_results(sc, cc, best_q_sc if err_split < err_joint else best_q_joint)

    # Save
    out_df = pd.DataFrame(our_rows)
    out_df.to_csv(DERIVED_DIR / 'monsted_fig3_our_computed.csv', index=False)
    print(f"\nSaved to {DERIVED_DIR / 'monsted_fig3_our_computed.csv'}")

    cmp_df = compare_with_transcription(our_rows, transcription_path)
    cmp_df.to_csv(DERIVED_DIR / 'monsted_fig3_comparison.csv', index=False)
    print(f"Saved to {DERIVED_DIR / 'monsted_fig3_comparison.csv'}")

    # Final pass/fail report
    print("\n=== PASS/FAIL vs thresholds ===")
    print(f"Panel A threshold: |delta| <= 1.0 pp")
    print(f"Panel B threshold: |delta| <= 1.0 counts")
    all_pass = True
    for row in cmp_df.itertuples(index=False):
        threshold = 1.0
        if abs(row.delta) > threshold:
            print(f"  FAIL  {row.panel},{row.model},k={row.k}: "
                  f"ours={row.ours:.5f} target={row.target} delta={row.delta:+.4f}")
            all_pass = False
    if all_pass:
        print("  ALL PASS")
    else:
        print(f"\nWorst row: {cmp_df.assign(abs_delta=cmp_df['delta'].abs()).sort_values('abs_delta').iloc[-1].to_dict()}")
