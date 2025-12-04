"""
Bulk-load Operation Outbreak simulations and Twitter cascades into the contagion database.

Runs with defaults; no CLI arguments required.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import sys
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_processing import create_tables
from src.data_processing.ORM import ContagionMetadata
from src.data_processing.database_utils import (
    insert_operation_outbreak_run,
    insert_twitter_run,
)
from src.data_processing.operation_outbreak import (
    compute_exposure_counts as compute_exposure_counts_oo,
    contact_adjacency,
    infection_time_series,
)
from src.data_processing.twitter_contagion import (
    DATA_DIR as TW_DATA_DIR,
    build_adjacency_for_participants,
    compute_exposure_counts as compute_exposure_counts_twitter,
    get_bot_ids,
    get_cascade_bot,
    get_cascade_hashtag,
    load_bot_tweets,
    load_retweet_cascades,
    retweet_time_series,
)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _run_exists(session, run_id: str) -> bool:
    return session.query(ContagionMetadata).filter_by(run_id=run_id).first() is not None


def _safe_iter(items: Iterable, limit: int | None):
    if limit is None:
        for item in items:
            yield item
    else:
        for idx, item in enumerate(items):
            if idx >= limit:
                break
            yield item


# ---------------------------------------------------------------------------
# Operation Outbreak bulk load
# ---------------------------------------------------------------------------

def load_all_oo(
    session,
    histories: pd.DataFrame,
    infections: pd.DataFrame,
    *,
    weight_type: str,
    freq: str,
    directed: bool,
    include_exposures: bool,
    overwrite: bool,
):
    sim_ids = sorted(set(histories["sim_id"].unique()) & set(infections["sim_id"].unique()))
    print(f"[OO] {len(sim_ids)} simulations with both histories and infections.")

    for sim_id in sim_ids:
        run_id = f"oo_sim_{sim_id}_{weight_type}"
        if not overwrite and _run_exists(session, run_id):
            print(f"[OO][skip] {run_id} already in DB")
            continue

        sim_hist = histories[histories["sim_id"] == sim_id]
        sim_inf = infections[infections["sim_id"] == sim_id]
        if sim_hist.empty or sim_inf.empty:
            print(f"[OO][warn] missing data for sim {sim_id}; histories rows={len(sim_hist)}, infections rows={len(sim_inf)}")
            continue

        print(f"[OO][{sim_id}] build adjacency (weight={weight_type}, directed={directed})")
        adjacency = contact_adjacency(sim_hist, sim_id, weight=weight_type, directed=directed)
        print(f"[OO][{sim_id}] adjacency {adjacency.shape}, edges {(adjacency > 0).sum().sum()}")

        print(f"[OO][{sim_id}] build infection time series (freq={freq})")
        ts = infection_time_series(sim_inf, sim_id, freq=freq)
        print(f"[OO][{sim_id}] timeseries {ts.shape}")

        exposure_M = exposure_N = None
        if include_exposures:
            print(f"[OO][{sim_id}] build exposure counts (kmax=8)")
            exposure_M, exposure_N = compute_exposure_counts_oo(
                infections=sim_inf,
                sim_id=sim_id,
                adjacency=adjacency,
                freq=freq,
                participants=adjacency.index,
                kmax=8,
            )

        print(f"[OO][{sim_id}] insert run_id={run_id}")
        insert_operation_outbreak_run(
            session,
            run_id=run_id,
            sim_id=sim_id,
            adjacency=adjacency,
            timeseries=ts,
            timeseries_freq=freq,
            weight_type=weight_type,
            is_directed=directed,
            description=f"Operation Outbreak sim {sim_id} ({weight_type})",
            exposure_success=exposure_M,
            exposure_failure=exposure_N,
        )
        print(f"[OO][{sim_id}] ok")


# ---------------------------------------------------------------------------
# Twitter bulk load
# ---------------------------------------------------------------------------

def load_all_twitter(
    session,
    edges_df: pd.DataFrame,
    retweets: list,
    bot_tweets: list,
    *,
    freq: str,
    include_source: bool,
    include_exposures: bool,
    overwrite: bool,
    limit: int | None,
):
    bot_ids = get_bot_ids()
    cascade_sizes = [(cid, len(cascade)) for cid, cascade in enumerate(retweets)]
    cascade_sizes.sort(key=lambda x: x[1], reverse=True)
    print(f"[TW] {len(cascade_sizes)} cascades; processing {limit or 'all'} ordered by size desc.")

    for cid, _ in _safe_iter(cascade_sizes, limit):
        run_id = f"twitter_cascade_{cid}"
        if not overwrite and _run_exists(session, run_id):
            print(f"[TW][skip] {run_id} already in DB")
            continue

        cascade = retweets[cid]
        if not cascade:
            print(f"[TW][warn] cascade {cid} empty; skipping")
            continue

        participants = {t["user"]["id"] for t in cascade}

        try:
            adjacency = build_adjacency_for_participants(edges_df, participants, include_bots=True)
        except Exception as exc:
            print(f"[TW][warn] cascade {cid} adjacency failed: {exc}")
            continue

        try:
            ts = retweet_time_series(
                retweets,
                cid,
                freq=freq,
                participants=adjacency.index,
                include_source=include_source,
                bot_tweets=bot_tweets,
            )
        except Exception as exc:
            print(f"[TW][warn] cascade {cid} timeseries failed: {exc}")
            continue

        exposure_M = exposure_N = None
        if include_exposures:
            try:
                exposure_M, exposure_N = compute_exposure_counts_twitter(
                    retweets=retweets,
                    cascade_id=cid,
                    adjacency=adjacency,
                    bot_tweets=bot_tweets,
                    freq=freq,
                    include_source=include_source,
                    participants=adjacency.index,
                    kmax=8,
                )
            except Exception as exc:
                print(f"[TW][warn] cascade {cid} exposure counts failed: {exc}")
                exposure_M = exposure_N = None

        hashtag = get_cascade_hashtag(retweets, cid, bot_tweets)
        source_bot = get_cascade_bot(retweets, cid, bot_tweets)
        original_tweet_id = cascade[0].get("retweeted_status", {}).get("id") if cascade else None

        print(f"[TW][{cid}] adj {adjacency.shape}, ts {ts.shape}, hashtag #{hashtag}")

        insert_twitter_run(
            session,
            run_id=run_id,
            cascade_id=cid,
            adjacency=adjacency,
            timeseries=ts,
            timeseries_freq=freq,
            intervention_hashtag=hashtag,
            source_bot_id=str(source_bot) if source_bot is not None else None,
            original_tweet_id=str(original_tweet_id) if original_tweet_id is not None else None,
            include_source=include_source,
            is_directed=True,
            description=f"Twitter cascade {cid} (# {hashtag})",
            extra_metadata=None,
            bot_ids=bot_ids,
            exposure_success=exposure_M,
            exposure_failure=exposure_N,
        )
        print(f"[TW][{cid}] ok")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    db_path = Path("data/contagion.db")
    oo_data_dir = Path("data") / "Blanchard-Thesis-main" / "data"
    tw_data_dir = TW_DATA_DIR

    # Defaults
    oo_weight_type = "duration"
    oo_freq = "1H"
    oo_directed = False

    tw_freq = "6H"
    tw_include_source = True
    tw_limit = None  # process all cascades

    include_exposures = True
    overwrite = False

    print(f"Using database: {db_path}")
    engine = create_engine(f"sqlite:///{db_path}")
    create_tables(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Operation Outbreak
        histories_path = oo_data_dir / "oo-histories-all.csv"
        infections_path = oo_data_dir / "infections_all.csv"
        if histories_path.exists() and infections_path.exists():
            print(f"\n[OO] Loading data from {oo_data_dir}")
            histories = pd.read_csv(histories_path, low_memory=False)
            infections = pd.read_csv(infections_path, low_memory=False)
            load_all_oo(
                session,
                histories,
                infections,
                weight_type=oo_weight_type,
                freq=oo_freq,
                directed=oo_directed,
                include_exposures=include_exposures,
                overwrite=overwrite,
            )
        else:
            print(f"[OO] Skipping: missing CSVs in {oo_data_dir}")

        # Twitter
        edges_path = tw_data_dir / "edges"
        retweets_path = tw_data_dir / "retweets_"
        bot_tweets_path = tw_data_dir / "all_bot_tweets_"
        if edges_path.exists() and retweets_path.exists() and bot_tweets_path.exists():
            print(f"\n[TW] Loading data from {tw_data_dir}")
            edges_df = pd.read_csv(edges_path, low_memory=False)
            edges_df["account_id"] = pd.to_numeric(edges_df["account_id"], errors="coerce")
            edges_df["bot_id"] = pd.to_numeric(edges_df["bot_id"], errors="coerce")
            edges_df = edges_df.dropna(subset=["account_id", "bot_id"]).astype({"account_id": "Int64", "bot_id": "Int64"})

            retweets = load_retweet_cascades(filename=retweets_path)
            bot_tweets = load_bot_tweets(filename=bot_tweets_path)

            load_all_twitter(
                session,
                edges_df,
                retweets,
                bot_tweets,
                freq=tw_freq,
                include_source=tw_include_source,
                include_exposures=include_exposures,
                overwrite=overwrite,
                limit=tw_limit,
            )
        else:
            print(f"[TW] Skipping: missing files in {tw_data_dir}")
    finally:
        session.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
