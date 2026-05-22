"""
Monsted Figure Verification Script
===================================
Independently verifies whether replicated exposure->adoption data matches the
Monsted et al. figure (monstead-results.png).

Runs two pipelines side-by-side for q=1.0:
  (A) WITH active_user_ids filter  -- current process_twitter.py behaviour
  (B) WITHOUT active_user_ids filter -- original author's exact logic

Transcribed Monsted figure values (Panel A, "Observed" bars, q=1.0):
  Simple contagion k vs % retweeted (read from figure by eye, ±0.002):
    k=1 ~0.010, k=2 ~0.013, k=3 ~0.020, k=4 ~0.022, k=5 ~0.022, k=6 ~0.023, k=7 ~0.025
  Complex contagion kappa vs % retweeted:
    kap=1 ~0.010, kap=2 ~0.012, kap=3 ~0.018, kap=4 ~0.020, kap=5 ~0.027, kap=6 ~0.030,
    kap=7 ~0.033, kap=8 ~0.040

Usage:
    uv run python scripts/verify_monsted_replication.py
"""

from __future__ import division
import sys
import json
import sqlite3
import datetime
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "twitter_contagion_data"
sys.path.insert(0, str(DATA_DIR))  # for the author's tools.py if needed

# ── Monsted figure transcription (Panel A, "Observed" bars, q=1.0) ──────────
# Read by eye from monstead-results.png; uncertainty ~±0.003
MONSTED_SC = {1: 0.010, 2: 0.013, 3: 0.020, 4: 0.022, 5: 0.022, 6: 0.023, 7: 0.025}
MONSTED_CC = {1: 0.010, 2: 0.012, 3: 0.018, 4: 0.020, 5: 0.027, 6: 0.030, 7: 0.033, 8: 0.040}

# ── Bot IDs (from tools.py) ──────────────────────────────────────────────────
BOT_ID2HANDLE = {
    2749655899: 'FitVeganGirl_', 2786086363: 'AndrenatorC', 2786246510: 'Van_Jester',
    2786363222: 'SimonWJorgensen', 2786427865: 'SuperRexy', 2787456828: 'matsie_at_dtu',
    2787463746: 'TheRexyGuy', 2787465462: 'RealAndersDuck', 2787474126: 'zakflanigan',
    2787481304: 'Pralesworth', 2787482192: 'marcussor', 2787485774: 'ioapsy',
    2787485978: 'Awe5omeMike', 2787486036: 'AnnasHollywood', 2787486060: 'sonia_manning',
    2787489548: 'ethanwoods88', 2787498932: 'LakerolIs', 2787499142: 'madpopo79',
    2787499932: 'EmiiiSuu', 2787500612: 'AxelCyrilian', 2787503576: 'THINKDEEPYO',
    2787514615: 'Shtinoehh', 2787516307: 'SirZenji', 2787527221: 'RichardsIndie',
    2787531548: 'CPH_Startup', 2787540506: 'canuckWong', 2787554990: 'SusanooGirl',
    2787610574: 'AlyciaGerald', 2787613616: 'MyNikkles', 2787620990: 'meetjamesmet',
    2787739466: 'tfosterm', 2787833887: 'ClintCrock', 2787905696: 'JackBoHorseMan',
    2788164512: 'hirihiker', 2795790799: 'hybrishybris', 2801335412: 'jsmth_t',
    2802511814: 'neergdave', 2829847496: 'ericfullhammer', 2841459034: 'zoesprings',
}
BOT_IDS = set(BOT_ID2HANDLE.keys())

INTERVENTIONS = [
    'getyourflushot', 'highfiveastranger', 'somethinggood',
    'HowManyPushups', 'turkeyface', 'SFThanks', 'blackfridaystories', 'BanksySF',
]


def parse_created_at(s):
    return datetime.datetime.strptime(s, '%a %b %d %H:%M:%S +0000 %Y')


def load_bot_tweets():
    """Load all original bot tweets (not retweets)."""
    path = DATA_DIR / "all_bot_tweets_"
    with open(path, 'r') as f:
        raw = json.load(f)
    return {t['id']: t for t in raw if 'retweeted_status' not in t}


def load_retweets():
    """Load ID2retweets: bot_tweet_id -> list of retweet objects."""
    path = DATA_DIR / "retweets_"
    with open(path, 'r') as f:
        raw = json.load(f)

    ID2retweets = {}
    if isinstance(raw, list):
        for tweet_group in raw:
            if not tweet_group:
                continue
            orig_id = tweet_group[0].get('retweeted_status', {}).get('id')
            if orig_id is None:
                continue
            if orig_id not in ID2retweets:
                ID2retweets[orig_id] = []
            ID2retweets[orig_id].extend(tweet_group)
    elif isinstance(raw, dict):
        for k, v in raw.items():
            ID2retweets[int(k)] = v

    # Sort each list by created_at
    for k in ID2retweets:
        ID2retweets[k].sort(key=lambda t: parse_created_at(t['created_at']))
    return ID2retweets


def build_graph():
    """Build bipartite bot-user graph from edges CSV + targets.db."""
    G = nx.Graph()

    # Load targets
    conn = sqlite3.connect(DATA_DIR / "targets.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id_str, handle FROM targets")
    targets = {int(row[0]): row[1] for row in cursor}
    conn.close()

    # Load edges
    edges_df = pd.read_csv(DATA_DIR / "edges", low_memory=False)
    for _, row in edges_df.iterrows():
        try:
            acct = int(row['account_id'])
            bot = int(row['bot_id'])
        except (ValueError, TypeError):
            continue
        G.add_edge(acct, bot)

    # Tag account types
    for node in G.nodes():
        if node in BOT_IDS:
            G.nodes[node]['account_type'] = 'bot'
        else:
            G.nodes[node]['account_type'] = 'user'

    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  User nodes: {sum(1 for n in G.nodes() if G.nodes[n].get('account_type')=='user'):,}")
    return G


def run_pipeline(G, ID2bot_tweet, ID2retweets, active_user_ids=None, label=""):
    """
    Core pipeline: author's exact logic, optionally with active_user_ids filter.
    
    Returns: dict with 'M_k', 'N_k', 'M_kappa', 'N_kappa'
             each is Counter(k -> count), aggregated across all interventions.
    """
    M_k = Counter()
    N_k = Counter()
    M_kappa = Counter()
    N_kappa = Counter()

    # Sort tweets by intervention
    intervention2tweets = {intv: [] for intv in INTERVENTIONS}
    for tid, tweet in ID2bot_tweet.items():
        hashtags = [h['text'] for h in tweet['entities']['hashtags']]
        matches = [intv for intv in INTERVENTIONS
                   if any(intv.lower() == h.lower() for h in hashtags)]
        if len(matches) == 1:
            intervention2tweets[matches[0]].append(tweet)

    intervention2user_rts = {intv: [] for intv in INTERVENTIONS}
    intervention2bot_rts = {intv: [] for intv in INTERVENTIONS}
    for intv, tweets in intervention2tweets.items():
        for tweet in tweets:
            tid = tweet['id']
            if tid not in ID2retweets:
                continue
            for rt in ID2retweets[tid]:
                if rt['user']['id'] in BOT_IDS:
                    intervention2bot_rts[intv].append(rt)
                else:
                    intervention2user_rts[intv].append(rt)

    # Ban top 2.5% retweeters
    all_rts = [rt for rts in intervention2user_rts.values() for rt in rts]
    user2rt_counts = Counter(rt['user']['id'] for rt in all_rts)
    if user2rt_counts:
        rt_max = np.percentile(list(user2rt_counts.values()), 97.5)
        banned = {k for k, v in user2rt_counts.items() if v > rt_max}
        for n in banned:
            if n in G.nodes:
                G.nodes[n]['account_type'] = 'banned'
    else:
        banned = set()

    print(f"  [{label}] Banned {len(banned)} suspected bots")

    for intv in INTERVENTIONS:
        # First retweet time per user
        user2first_rt = {}
        for rt in intervention2user_rts[intv]:
            uid = rt['user']['id']
            t = parse_created_at(rt['created_at'])
            if uid not in user2first_rt or t < user2first_rt[uid]:
                user2first_rt[uid] = t

        # Exposure tracking: k (tweet IDs) and kappa (unique bots)
        user2tweet_ids = {}   # for k metric
        user2bot_ids = {}     # for kappa metric

        def expose(user, tweet_id, bot_id, tweet_time):
            # Apply active filter if requested
            if active_user_ids is not None and user not in active_user_ids:
                return
            if G.nodes.get(user, {}).get('account_type') != 'user':
                return
            # Skip if exposure after adoption
            if user in user2first_rt and user2first_rt[user] < tweet_time:
                return
            user2tweet_ids.setdefault(user, set()).add(tweet_id)
            user2bot_ids.setdefault(user, set()).add(bot_id)

        # Original bot tweets
        for tweet in intervention2tweets[intv]:
            origin_bot = tweet['user']['id']
            tweet_id = tweet['id']
            tweet_time = parse_created_at(tweet['created_at'])
            if origin_bot not in G:
                continue
            for user in G.neighbors(origin_bot):
                expose(user, tweet_id, origin_bot, tweet_time)

        # Bot retweets (count_retweets_as_exposures=True)
        for rt in intervention2bot_rts[intv]:
            proxy_bot = rt['user']['id']
            origin_bot = rt['retweeted_status']['user']['id']
            orig_tweet_id = rt['retweeted_status']['id']
            rt_time = parse_created_at(rt['created_at'])
            if proxy_bot not in G:
                continue
            for user in G.neighbors(proxy_bot):
                expose(user, orig_tweet_id, origin_bot, rt_time)

        # Accumulate M/N
        for user in user2tweet_ids:
            k_val = len(user2tweet_ids[user])          # simple contagion
            kap_val = len(user2bot_ids.get(user, []))  # complex contagion
            adopted = user in user2first_rt
            if adopted:
                M_k[k_val] += 1
                M_kappa[kap_val] += 1
            else:
                N_k[k_val] += 1
                N_kappa[kap_val] += 1

    return {'M_k': M_k, 'N_k': N_k, 'M_kappa': M_kappa, 'N_kappa': N_kappa}


def adoption_rates(M, N, bins):
    """Return list of (bin, M, N, total, P_adopt) for each bin."""
    rows = []
    for b in bins:
        m = M.get(b, 0)
        n = N.get(b, 0)
        tot = m + n
        p = m / tot if tot > 0 else float('nan')
        rows.append((b, m, n, tot, p))
    return rows


def print_comparison_table(label, rows_with, rows_without, monsted_ref):
    print(f"\n{'='*90}")
    print(f"  {label}")
    print(f"{'='*90}")
    print(f"{'k':>4} | {'WITH filter':^30} | {'WITHOUT filter (author)':^30} | {'Monsted fig':^12} | Ratio")
    print(f"     | {'M':>5} {'N':>6} {'tot':>6} {'P%':>6} | {'M':>5} {'N':>6} {'tot':>6} {'P%':>6} | {'P%':>8}    | (no-filt/Mnstd)")
    print(f"{'-'*90}")
    for (b, mw, nw, tw, pw), (b2, mwo, nwo, two, pwo) in zip(rows_with, rows_without):
        ref = monsted_ref.get(b, float('nan'))
        ratio = (pwo / ref) if (not np.isnan(pwo) and not np.isnan(ref) and ref > 0) else float('nan')
        pw_str  = f"{pw*100:5.2f}" if not np.isnan(pw) else "  nan"
        pwo_str = f"{pwo*100:5.2f}" if not np.isnan(pwo) else "  nan"
        ref_str = f"{ref*100:5.2f}" if not np.isnan(ref) else "  nan"
        rat_str = f"{ratio:5.2f}x" if not np.isnan(ratio) else "  nan"
        print(f"{b:4d} | {mw:5.0f} {nw:6.0f} {tw:6.0f} {pw_str}% | {mwo:5.0f} {nwo:6.0f} {two:6.0f} {pwo_str}% | {ref_str}%    | {rat_str}")


def main():
    print("=" * 70)
    print("Monsted Figure Verification")
    print("=" * 70)

    print("\nStep 1: Loading data...")
    ID2bot_tweet = load_bot_tweets()
    print(f"  Bot tweets: {len(ID2bot_tweet):,}")
    ID2retweets = load_retweets()
    print(f"  Retweet groups: {len(ID2retweets):,}")

    print("\nStep 2: Building graph...")
    G = build_graph()

    print("\nStep 3: Identifying active users (retweet filter)...")
    active_user_ids = set()
    for rts in ID2retweets.values():
        for rt in rts:
            active_user_ids.add(rt['user']['id'])
    print(f"  Active users (ever retweeted): {len(active_user_ids):,}")

    print("\nStep 4: Running WITH active filter (current implementation)...")
    res_with = run_pipeline(G, ID2bot_tweet, ID2retweets,
                            active_user_ids=active_user_ids, label="WITH filter")

    print("\nStep 5: Running WITHOUT active filter (original author logic)...")
    res_without = run_pipeline(G, ID2bot_tweet, ID2retweets,
                               active_user_ids=None, label="WITHOUT filter")

    # ── Summary stats ─────────────────────────────────────────────────────────
    print("\n── Population summary ──────────────────────────────────────────────")
    for label, res in [("WITH filter", res_with), ("WITHOUT filter", res_without)]:
        tot_m = sum(res['M_k'].values())
        tot_n = sum(res['N_k'].values())
        p_ov = tot_m / (tot_m + tot_n) if (tot_m + tot_n) > 0 else 0
        print(f"  {label:<28}: adopters={tot_m:6,}  non-adopters={tot_n:7,}  "
              f"total={tot_m+tot_n:7,}  P(adopt)={p_ov*100:.2f}%")

    # ── Detailed comparison tables ─────────────────────────────────────────────
    k_bins = list(range(1, 8))
    kap_bins = list(range(1, 9))

    rows_k_with    = adoption_rates(res_with['M_k'],    res_with['N_k'],    k_bins)
    rows_k_without = adoption_rates(res_without['M_k'], res_without['N_k'], k_bins)

    rows_kap_with    = adoption_rates(res_with['M_kappa'],    res_with['N_kappa'],    kap_bins)
    rows_kap_without = adoption_rates(res_without['M_kappa'], res_without['N_kappa'], kap_bins)

    print_comparison_table("Simple Contagion — k metric", rows_k_with, rows_k_without, MONSTED_SC)
    print_comparison_table("Complex Contagion — kappa metric", rows_kap_with, rows_kap_without, MONSTED_CC)

    # ── Pass/Fail summary ──────────────────────────────────────────────────────
    print("\n── Pass/Fail: |P_no_filter - P_Monsted| / P_Monsted  (threshold 20%) ──")
    THRESHOLD = 0.20
    all_pass = True
    for label, rows, ref_dict in [
        ("SC k", rows_k_without, MONSTED_SC),
        ("CC kappa", rows_kap_without, MONSTED_CC),
    ]:
        print(f"\n  {label}")
        for (b, m, n, tot, p) in rows:
            ref = ref_dict.get(b, float('nan'))
            if np.isnan(p) or np.isnan(ref) or ref == 0:
                status = "SKIP (no data)"
            else:
                rel_err = abs(p - ref) / ref
                if rel_err <= THRESHOLD:
                    status = f"PASS  (rel_err={rel_err*100:.1f}%)"
                else:
                    status = f"FAIL  (rel_err={rel_err*100:.1f}%)"
                    all_pass = False
            print(f"    k={b}: P_no_filt={p*100:.2f}%  P_monsted={ref*100:.2f}%  -> {status}")

    print(f"\n{'='*70}")
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL — see mismatch table above'}")
    print(f"{'='*70}")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    out_rows = []
    for (b, mw, nw, tw, pw), (b2, mwo, nwo, two, pwo) in zip(rows_k_with, rows_k_without):
        ref = MONSTED_SC.get(b, float('nan'))
        out_rows.append({
            'metric': 'k', 'bin': b,
            'M_with_filter': mw, 'N_with_filter': nw, 'P_with_filter': pw,
            'M_no_filter': mwo, 'N_no_filter': nwo, 'P_no_filter': pwo,
            'P_monsted': ref,
            'rel_err_pct': abs(pwo - ref) / ref * 100 if not np.isnan(pwo) and ref > 0 else float('nan'),
        })
    for (b, mw, nw, tw, pw), (b2, mwo, nwo, two, pwo) in zip(rows_kap_with, rows_kap_without):
        ref = MONSTED_CC.get(b, float('nan'))
        out_rows.append({
            'metric': 'kappa', 'bin': b,
            'M_with_filter': mw, 'N_with_filter': nw, 'P_with_filter': pw,
            'M_no_filter': mwo, 'N_no_filter': nwo, 'P_no_filter': pwo,
            'P_monsted': ref,
            'rel_err_pct': abs(pwo - ref) / ref * 100 if not np.isnan(pwo) and ref > 0 else float('nan'),
        })

    out_path = PROJECT_ROOT / "results" / "monsted_verification.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(out_path, index=False)
    print(f"\nDetailed results saved to: {out_path}")


if __name__ == "__main__":
    main()
