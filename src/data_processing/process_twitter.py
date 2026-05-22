"""
Twitter Contagion Data Processing - Verified Implementation

This script uses the proven exposure counting logic from the author
reproduction code and extends it to compute:
- M/N matrices for k (simple contagion)
- M/N matrices for kappa (complex contagion)
- A-matrices (exposure counts per bot)
- B-vectors (adoption indicators)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import utils
except (ImportError, ValueError):
    import utils


DEFAULT_DATA_DIR = Path("data/twitter_contagion_data").resolve()
DEFAULT_PROCESSED_DIR = Path("data/processed/twitter").resolve()
DEFAULT_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class TwitterPaths:
    data_dir: Path
    processed_dir: Path


@dataclass(frozen=True)
class TwitterProcessingInputs:
    graph: object
    interventions: tuple[str, ...]
    intervention_tweets: dict[str, list[dict]]
    intervention_user_retweets: dict[str, list[dict]]
    intervention_bot_retweets: dict[str, list[dict]]


def _resolve_paths(
    data_dir: str | Path | None = None,
    output_base_dir: str | Path | None = None,
) -> TwitterPaths:
    resolved_data_dir = Path(data_dir).resolve() if data_dir is not None else DEFAULT_DATA_DIR
    resolved_processed_dir = (
        Path(output_base_dir).resolve() if output_base_dir is not None else DEFAULT_PROCESSED_DIR
    )
    return TwitterPaths(data_dir=resolved_data_dir, processed_dir=resolved_processed_dir)


def _resolve_q_values(q=1.0, q_sc=None, q_cc=None) -> tuple[float, float]:
    resolved_q_sc = q_sc if q_sc is not None else q
    resolved_q_cc = q_cc if q_cc is not None else q
    return resolved_q_sc, resolved_q_cc


def _match_intervention(hashtags: list[str], interventions: tuple[str, ...]) -> str | None:
    matches = [name for name in interventions if any(name.lower() == tag.lower() for tag in hashtags)]
    if len(matches) != 1:
        return None
    return matches[0]


def _build_intervention_tweets(
    id_to_bot_tweet: dict[int, dict],
    interventions: tuple[str, ...],
) -> dict[str, list[dict]]:
    intervention_tweets = {intervention: [] for intervention in interventions}
    for tweet in id_to_bot_tweet.values():
        hashtags = [item["text"] for item in tweet["entities"]["hashtags"]]
        intervention = _match_intervention(hashtags, interventions)
        if intervention is not None:
            intervention_tweets[intervention].append(tweet)
    return intervention_tweets


def _split_retweets_by_intervention(
    intervention_tweets: dict[str, list[dict]],
    id_to_retweets: dict[int, list[dict]],
    bot_ids: set[int],
) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    intervention_user_retweets = {intervention: [] for intervention in intervention_tweets}
    intervention_bot_retweets = {intervention: [] for intervention in intervention_tweets}

    for intervention, tweets in intervention_tweets.items():
        for tweet in tweets:
            for retweet in id_to_retweets.get(tweet["id"], []):
                if retweet["user"]["id"] in bot_ids:
                    intervention_bot_retweets[intervention].append(retweet)
                else:
                    intervention_user_retweets[intervention].append(retweet)

    return intervention_user_retweets, intervention_bot_retweets


def _mark_banned_accounts(graph, intervention_user_retweets: dict[str, list[dict]]) -> None:
    all_retweets = [retweet for retweets in intervention_user_retweets.values() for retweet in retweets]
    if not all_retweets:
        return

    user_to_retweet_counts = Counter(retweet["user"]["id"] for retweet in all_retweets)
    retweet_cutoff = np.percentile(list(user_to_retweet_counts.values()), 97.5)

    for user_id, retweet_count in user_to_retweet_counts.items():
        if retweet_count > retweet_cutoff and user_id in graph:
            graph.nodes[user_id]["account_type"] = "banned"


def load_processing_inputs(data_dir: str | Path | None = None) -> TwitterProcessingInputs:
    resolved_data_dir = _resolve_paths(data_dir=data_dir).data_dir

    print("Loading data using author's utils...")
    id_to_bot_tweet = utils.get_ID2bot_tweet(data_dir=resolved_data_dir)
    id_to_retweets = utils.get_ID2retweets(data_dir=resolved_data_dir)
    interventions = tuple(utils.interventions)
    bot_ids = utils.bot_IDs

    intervention_tweets = _build_intervention_tweets(id_to_bot_tweet, interventions)
    intervention_user_retweets, intervention_bot_retweets = _split_retweets_by_intervention(
        intervention_tweets,
        id_to_retweets,
        bot_ids,
    )

    print("Building graph...")
    graph = utils.make_graph(data_dir=resolved_data_dir)
    _mark_banned_accounts(graph, intervention_user_retweets)

    return TwitterProcessingInputs(
        graph=graph,
        interventions=interventions,
        intervention_tweets=intervention_tweets,
        intervention_user_retweets=intervention_user_retweets,
        intervention_bot_retweets=intervention_bot_retweets,
    )


def _first_retweet_times(user_retweets: list[dict]) -> dict[int, object]:
    user_to_first_retweet = {}
    for retweet in user_retweets:
        user_id = retweet["user"]["id"]
        retweet_time = utils.parse_created_at(retweet["created_at"])
        if user_id not in user_to_first_retweet or retweet_time < user_to_first_retweet[user_id]:
            user_to_first_retweet[user_id] = retweet_time
    return user_to_first_retweet


def _is_valid_exposure_target(graph, user, event_time, user_to_first_retweet) -> bool:
    if user in user_to_first_retweet and user_to_first_retweet[user] < event_time:
        return False
    return graph.nodes[user].get("account_type") == "user"


def _accumulate_exposure_event(
    *,
    graph,
    broadcaster_id,
    origin_tweet_id,
    origin_bot_id,
    event_time,
    user_to_first_retweet,
    user_to_exposed_tweets,
    user_to_bot_tweets,
) -> None:
    if broadcaster_id not in graph:
        return

    for user in graph.neighbors(broadcaster_id):
        if not _is_valid_exposure_target(graph, user, event_time, user_to_first_retweet):
            continue

        user_to_exposed_tweets.setdefault(user, []).append(origin_tweet_id)
        user_to_bot_tweets.setdefault(user, {}).setdefault(origin_bot_id, []).append(origin_tweet_id)


def _build_user_records(
    user_to_exposed_tweets: dict,
    user_to_bot_tweets: dict,
    user_to_first_retweet: dict,
    *,
    q_sc: float,
    q_cc: float,
) -> list[dict]:
    user_data = []
    for user, exposed_tweets in user_to_exposed_tweets.items():
        n_raw = len(set(exposed_tweets))
        p_dist_k = utils.compute_binomial_distribution(n_raw, q_sc)

        bot_exposures = {
            bot_id: len(tweets)
            for bot_id, tweets in user_to_bot_tweets[user].items()
        }
        p_dist_kappa = utils.compute_exposure_probabilities_fast(list(bot_exposures.values()), q_cc)

        user_data.append(
            {
                "user_id": str(user),
                "p_dist_k": p_dist_k,
                "p_dist_kappa": p_dist_kappa,
                "adopted": user in user_to_first_retweet,
                "bot_exposures": bot_exposures,
            }
        )
    return user_data


def process_intervention(
    intervention,
    G,
    intervention2tweets,
    intervention2user_retweets,
    intervention2bot_retweets,
    bot_IDs,
    q=1.0,
    q_sc=None,
    q_cc=None,
    count_retweets_as_exposures=True,
):
    """
    Process a single intervention using verified author logic.

    Returns user-level data with:
    - k: number of unique tweet IDs (simple contagion)
    - kappa: number of unique bot sources (complex contagion)
    - bot_exposures: dict mapping bot_id -> count
    - adopted: boolean
    """
    del bot_IDs

    print(f"Processing {intervention}...")
    resolved_q_sc, resolved_q_cc = _resolve_q_values(q=q, q_sc=q_sc, q_cc=q_cc)

    user_to_first_retweet = _first_retweet_times(intervention2user_retweets[intervention])
    user_to_exposed_tweets = {}
    user_to_bot_tweets = {}

    for tweet in intervention2tweets[intervention]:
        _accumulate_exposure_event(
            graph=G,
            broadcaster_id=tweet["user"]["id"],
            origin_tweet_id=tweet["id"],
            origin_bot_id=tweet["user"]["id"],
            event_time=utils.parse_created_at(tweet["created_at"]),
            user_to_first_retweet=user_to_first_retweet,
            user_to_exposed_tweets=user_to_exposed_tweets,
            user_to_bot_tweets=user_to_bot_tweets,
        )

    for tweet in intervention2bot_retweets[intervention]:
        proxy_bot = tweet["user"]["id"]
        origin_bot = (
            tweet["retweeted_status"]["user"]["id"]
            if count_retweets_as_exposures
            else proxy_bot
        )
        _accumulate_exposure_event(
            graph=G,
            broadcaster_id=proxy_bot,
            origin_tweet_id=tweet["retweeted_status"]["id"],
            origin_bot_id=origin_bot,
            event_time=utils.parse_created_at(tweet["created_at"]),
            user_to_first_retweet=user_to_first_retweet,
            user_to_exposed_tweets=user_to_exposed_tweets,
            user_to_bot_tweets=user_to_bot_tweets,
        )

    return _build_user_records(
        user_to_exposed_tweets,
        user_to_bot_tweets,
        user_to_first_retweet,
        q_sc=resolved_q_sc,
        q_cc=resolved_q_cc,
    )


def _fold_prob_dist_to_kmax(p_dist, kmax):
    """
    Fold a distribution over exposure levels into 1..kmax bins.
    Mass at levels >= kmax is accumulated into bin kmax.
    """
    folded = np.zeros(kmax, dtype=float)
    for level, prob in p_dist.items():
        level_i = int(level)
        prob_f = float(prob)
        if level_i <= 0 or prob_f <= 0:
            continue
        idx = min(level_i, kmax) - 1
        folded[idx] += prob_f
    return folded


def _mn_rows_from_prob_dist(p_dist, adopted, kmax, mn_mode):
    """
    Build expected M/N row vectors for one user from a level distribution.

    Modes:
    - incidence: event mass placed at the terminal level only
    - survival: event at level nu with non-event mass at all prior at-risk levels
    """
    p_level = _fold_prob_dist_to_kmax(p_dist, kmax)
    M_vals = np.zeros(kmax, dtype=float)
    N_vals = np.zeros(kmax, dtype=float)

    if mn_mode == "incidence":
        if adopted:
            M_vals = p_level
        else:
            N_vals = p_level
        return M_vals, N_vals

    if mn_mode == "survival":
        suffix = np.flip(np.cumsum(np.flip(p_level)))
        if adopted:
            M_vals = p_level
            if kmax > 1:
                N_vals[:-1] = suffix[1:]
        else:
            N_vals = suffix
        return M_vals, N_vals

    raise ValueError(f"Unknown mn_mode='{mn_mode}'. Expected 'incidence' or 'survival'.")


def _level_row(user_id: str, values: np.ndarray) -> dict[str, float | str]:
    row = {"user_id": user_id}
    for idx, value in enumerate(values, start=1):
        row[str(idx)] = float(value)
    return row


def _write_metric_pair(
    output_dir: Path,
    *,
    metric: str,
    q_value: float,
    m_rows: list[dict],
    n_rows: list[dict],
) -> None:
    pd.DataFrame(m_rows).to_parquet(output_dir / f"M_{metric}_q{q_value}.parquet")
    pd.DataFrame(n_rows).to_parquet(output_dir / f"N_{metric}_q{q_value}.parquet")


def _write_ab_outputs(
    output_dir: Path,
    *,
    user_ids: list[str],
    bot_exposures_by_user: dict[str, dict[int, int] | Counter],
    adopted_by_user: dict[str, bool],
) -> tuple[int, int]:
    all_bot_ids = sorted(
        {bot_id for user_id in user_ids for bot_id in bot_exposures_by_user[user_id].keys()}
    )

    a_rows = []
    for user_id in user_ids:
        row = {"user_id": user_id}
        bot_counts = bot_exposures_by_user[user_id]
        for bot_id in all_bot_ids:
            row[str(bot_id)] = int(bot_counts.get(bot_id, 0))
        a_rows.append(row)
    pd.DataFrame(a_rows).to_parquet(output_dir / "A_matrix.parquet")

    b_rows = [{"user_id": user_id, "adopted": int(bool(adopted_by_user[user_id]))} for user_id in user_ids]
    pd.DataFrame(b_rows).to_parquet(output_dir / "B_vector.parquet")

    return len(user_ids), len(all_bot_ids)


def save_matrices(
    user_data,
    intervention,
    q=1.0,
    q_sc=None,
    q_cc=None,
    kmax=50,
    kappamax=20,
    mn_mode="incidence",
    processed_dir: str | Path | None = None,
):
    """q_sc overrides q for SC (k) files; q_cc overrides q for CC (kappa) files."""
    resolved_q_sc, resolved_q_cc = _resolve_q_values(q=q, q_sc=q_sc, q_cc=q_cc)
    mn_mode = str(mn_mode).lower()
    if mn_mode not in {"incidence", "survival"}:
        raise ValueError(f"mn_mode must be 'incidence' or 'survival'. Got: {mn_mode}")
    if not user_data:
        print(f"No exposure data for {intervention}")
        return

    output_root = _resolve_paths(output_base_dir=processed_dir).processed_dir
    output_dir = output_root / intervention
    output_dir.mkdir(parents=True, exist_ok=True)

    m_rows_k = []
    n_rows_k = []
    m_rows_kappa = []
    n_rows_kappa = []
    user_ids = []
    bot_exposures_by_user = {}
    adopted_by_user = {}

    for user in user_data:
        user_id = user["user_id"]
        adopted = bool(user["adopted"])
        user_ids.append(user_id)

        m_k_vals, n_k_vals = _mn_rows_from_prob_dist(
            user["p_dist_k"],
            adopted=adopted,
            kmax=kmax,
            mn_mode=mn_mode,
        )
        m_kappa_vals, n_kappa_vals = _mn_rows_from_prob_dist(
            user["p_dist_kappa"],
            adopted=adopted,
            kmax=kappamax,
            mn_mode=mn_mode,
        )

        m_rows_k.append(_level_row(user_id, m_k_vals))
        n_rows_k.append(_level_row(user_id, n_k_vals))
        m_rows_kappa.append(_level_row(user_id, m_kappa_vals))
        n_rows_kappa.append(_level_row(user_id, n_kappa_vals))
        bot_exposures_by_user[user_id] = user["bot_exposures"]
        adopted_by_user[user_id] = adopted

    _write_metric_pair(output_dir, metric="k", q_value=resolved_q_sc, m_rows=m_rows_k, n_rows=n_rows_k)
    _write_metric_pair(
        output_dir,
        metric="kappa",
        q_value=resolved_q_cc,
        m_rows=m_rows_kappa,
        n_rows=n_rows_kappa,
    )
    user_count, bot_count = _write_ab_outputs(
        output_dir,
        user_ids=user_ids,
        bot_exposures_by_user=bot_exposures_by_user,
        adopted_by_user=adopted_by_user,
    )

    print(f"Saved {user_count} users for {intervention} (mn_mode={mn_mode})")
    print(f"  - M/N_k_q{resolved_q_sc}.parquet (kmax={kmax})")
    print(f"  - M/N_kappa_q{resolved_q_cc}.parquet (kappamax={kappamax})")
    print(f"  - A_matrix.parquet ({bot_count} bots)")
    print("  - B_vector.parquet")


def save_full_outputs_from_aggregates(
    m_k_by_user,
    n_k_by_user,
    m_kappa_by_user,
    n_kappa_by_user,
    bot_exposures_by_user,
    adopted_by_user,
    q=1.0,
    q_sc=None,
    q_cc=None,
    kmax=50,
    kappamax=20,
    processed_dir: str | Path | None = None,
):
    """Write only full-level M/N/A/B outputs from in-memory user aggregates."""
    resolved_q_sc, resolved_q_cc = _resolve_q_values(q=q, q_sc=q_sc, q_cc=q_cc)
    output_root = _resolve_paths(output_base_dir=processed_dir).processed_dir
    output_dir = output_root / "full"
    output_dir.mkdir(parents=True, exist_ok=True)

    user_ids = sorted(set(adopted_by_user.keys()))
    if not user_ids:
        raise ValueError("No users found while building full-only Twitter outputs.")

    m_rows_k = [_level_row(user_id, m_k_by_user[user_id]) for user_id in user_ids]
    n_rows_k = [_level_row(user_id, n_k_by_user[user_id]) for user_id in user_ids]
    m_rows_kappa = [_level_row(user_id, m_kappa_by_user[user_id]) for user_id in user_ids]
    n_rows_kappa = [_level_row(user_id, n_kappa_by_user[user_id]) for user_id in user_ids]

    _write_metric_pair(output_dir, metric="k", q_value=resolved_q_sc, m_rows=m_rows_k, n_rows=n_rows_k)
    _write_metric_pair(
        output_dir,
        metric="kappa",
        q_value=resolved_q_cc,
        m_rows=m_rows_kappa,
        n_rows=n_rows_kappa,
    )
    user_count, bot_count = _write_ab_outputs(
        output_dir,
        user_ids=user_ids,
        bot_exposures_by_user=bot_exposures_by_user,
        adopted_by_user=adopted_by_user,
    )

    print(f"\nSaved full-only outputs to: {output_dir}")
    print(f"  - M/N_k_q{resolved_q_sc}.parquet (kmax={kmax})")
    print(f"  - M/N_kappa_q{resolved_q_cc}.parquet (kappamax={kappamax})")
    print(f"  - A_matrix.parquet ({bot_count} bots)")
    print(f"  - B_vector.parquet ({user_count} users)")


def aggregate_all_interventions(
    q=1.0,
    q_sc=None,
    q_cc=None,
    kmax=50,
    kappamax=20,
    processed_dir: str | Path | None = None,
):
    """
    Aggregate M/N matrices across all interventions using DuckDB.

    SC files use q_sc (defaults to q); CC files use q_cc (defaults to q).
    Saves aggregated results to data/processed/twitter/full/.
    """
    import duckdb

    del kmax, kappamax

    resolved_q_sc, resolved_q_cc = _resolve_q_values(q=q, q_sc=q_sc, q_cc=q_cc)
    output_root = _resolve_paths(output_base_dir=processed_dir).processed_dir
    output_dir = output_root / "full"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nAggregating interventions with q_sc={resolved_q_sc}, q_cc={resolved_q_cc}...")
    con = duckdb.connect()

    for metric, q_val in [("k", resolved_q_sc), ("kappa", resolved_q_cc)]:
        m_glob = str(output_root / f"*/M_{metric}_q{q_val}.parquet")
        n_glob = str(output_root / f"*/N_{metric}_q{q_val}.parquet")

        try:
            con.query(f"SELECT * FROM '{m_glob}' LIMIT 1").df()
        except Exception:
            print(f"  Warning: No data found for metric={metric}, q={q_val}")
            continue

        cols = con.query(f"SELECT * FROM '{m_glob}' LIMIT 0").df().columns
        exposure_cols = [column for column in cols if column not in ["user_id", "intervention"]]
        sum_expr = ", ".join([f'SUM("{column}") as "{column}"' for column in exposure_cols])

        m_df = con.query(
            f"""
            SELECT user_id, {sum_expr}
            FROM read_parquet('{m_glob}', filename=true)
            WHERE filename NOT LIKE '%/full/%'
            GROUP BY user_id ORDER BY user_id
            """
        ).df()

        n_df = con.query(
            f"""
            SELECT user_id, {sum_expr}
            FROM read_parquet('{n_glob}', filename=true)
            WHERE filename NOT LIKE '%/full/%'
            GROUP BY user_id ORDER BY user_id
            """
        ).df()

        m_df.to_parquet(output_dir / f"M_{metric}_q{q_val}.parquet")
        n_df.to_parquet(output_dir / f"N_{metric}_q{q_val}.parquet")
        print(f"  ✓ M/N_{metric}_q{q_val}.parquet: {len(m_df)} users, {len(exposure_cols)} exposure levels")

    con.close()
    print(f"\nAggregated data saved to: {output_dir}")


def build_exposure_matrices(
    data_dir=None,
    output_base_dir=None,
    q=1.0,
    q_sc=None,
    q_cc=None,
    count_retweets_as_exposures=True,
    mn_mode="incidence",
    full_only=False,
    kmax=50,
    kappamax=20,
):
    """
    Compatibility wrapper to maintain API for existing tests and scripts.
    q_sc overrides q for SC; q_cc overrides q for CC.
    """
    paths = _resolve_paths(data_dir=data_dir, output_base_dir=output_base_dir)
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    run_processing(
        data_dir=paths.data_dir,
        processed_dir=paths.processed_dir,
        q=q,
        q_sc=q_sc,
        q_cc=q_cc,
        count_retweets_as_exposures=count_retweets_as_exposures,
        mn_mode=mn_mode,
        full_only=full_only,
        kmax=kmax,
        kappamax=kappamax,
    )


def run_processing(
    data_dir=None,
    processed_dir=None,
    q=1.0,
    q_sc=None,
    q_cc=None,
    count_retweets_as_exposures=True,
    mn_mode="incidence",
    full_only=False,
    kmax=50,
    kappamax=20,
):
    """Main processing function using verified author logic."""
    print(f"M/N mode: {mn_mode}")
    print(f"Full-only mode: {full_only}")

    paths = _resolve_paths(data_dir=data_dir, output_base_dir=processed_dir)
    inputs = load_processing_inputs(paths.data_dir)

    m_k_by_user = {}
    n_k_by_user = {}
    m_kappa_by_user = {}
    n_kappa_by_user = {}
    bot_exposures_by_user = {}
    adopted_by_user = {}

    for intervention in inputs.interventions:
        user_data = process_intervention(
            intervention,
            inputs.graph,
            inputs.intervention_tweets,
            inputs.intervention_user_retweets,
            inputs.intervention_bot_retweets,
            bot_IDs=None,
            q=q,
            q_sc=q_sc,
            q_cc=q_cc,
            count_retweets_as_exposures=count_retweets_as_exposures,
        )
        if full_only:
            for user in user_data:
                user_id = user["user_id"]
                adopted = bool(user["adopted"])
                m_k_vals, n_k_vals = _mn_rows_from_prob_dist(
                    user["p_dist_k"],
                    adopted=adopted,
                    kmax=kmax,
                    mn_mode=mn_mode,
                )
                m_kappa_vals, n_kappa_vals = _mn_rows_from_prob_dist(
                    user["p_dist_kappa"],
                    adopted=adopted,
                    kmax=kappamax,
                    mn_mode=mn_mode,
                )

                if user_id not in m_k_by_user:
                    m_k_by_user[user_id] = np.zeros(kmax, dtype=float)
                    n_k_by_user[user_id] = np.zeros(kmax, dtype=float)
                    m_kappa_by_user[user_id] = np.zeros(kappamax, dtype=float)
                    n_kappa_by_user[user_id] = np.zeros(kappamax, dtype=float)
                    bot_exposures_by_user[user_id] = Counter()
                    adopted_by_user[user_id] = False

                m_k_by_user[user_id] += m_k_vals
                n_k_by_user[user_id] += n_k_vals
                m_kappa_by_user[user_id] += m_kappa_vals
                n_kappa_by_user[user_id] += n_kappa_vals
                bot_exposures_by_user[user_id].update(user["bot_exposures"])
                adopted_by_user[user_id] = adopted_by_user[user_id] or adopted
        else:
            save_matrices(
                user_data,
                intervention,
                q=q,
                q_sc=q_sc,
                q_cc=q_cc,
                kmax=kmax,
                kappamax=kappamax,
                mn_mode=mn_mode,
                processed_dir=paths.processed_dir,
            )

    if full_only:
        print("\n" + "=" * 70)
        print("WRITING FULL-ONLY OUTPUTS")
        print("=" * 70)
        save_full_outputs_from_aggregates(
            m_k_by_user=m_k_by_user,
            n_k_by_user=n_k_by_user,
            m_kappa_by_user=m_kappa_by_user,
            n_kappa_by_user=n_kappa_by_user,
            bot_exposures_by_user=bot_exposures_by_user,
            adopted_by_user=adopted_by_user,
            q=q,
            q_sc=q_sc,
            q_cc=q_cc,
            kmax=kmax,
            kappamax=kappamax,
            processed_dir=paths.processed_dir,
        )
    else:
        print("\n" + "=" * 70)
        print("AGGREGATING ALL INTERVENTIONS")
        print("=" * 70)
        aggregate_all_interventions(
            q=q,
            q_sc=q_sc,
            q_cc=q_cc,
            kmax=kmax,
            kappamax=kappamax,
            processed_dir=paths.processed_dir,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=float, default=1.0, help="Sensing probability (fallback for both SC and CC)")
    parser.add_argument("--q-sc", type=float, default=None, help="Sensing probability for SC (overrides --q)")
    parser.add_argument("--q-cc", type=float, default=None, help="Sensing probability for CC (overrides --q)")
    parser.add_argument("--count_retweets", action="store_true", default=True, help="Count retweets as exposures")
    parser.add_argument("--data-dir", type=str, help="Input data directory")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--no-bot-retweets", action="store_true", help="Deprecated/Ignored")
    parser.add_argument(
        "--mn-mode",
        type=str,
        default="incidence",
        choices=["incidence", "survival"],
        help="How to build M/N matrices from exposure distributions.",
    )
    parser.add_argument(
        "--full-only",
        action="store_true",
        help="Write only aggregated full outputs (skip per-intervention outputs).",
    )
    parser.add_argument("--kmax", type=int, default=50, help="Maximum exposure bin for k.")
    parser.add_argument("--kappamax", type=int, default=20, help="Maximum exposure bin for kappa.")

    args = parser.parse_args()

    build_exposure_matrices(
        data_dir=args.data_dir,
        output_base_dir=args.output_dir,
        q=args.q,
        q_sc=args.q_sc,
        q_cc=args.q_cc,
        count_retweets_as_exposures=args.count_retweets,
        mn_mode=args.mn_mode,
        full_only=args.full_only,
        kmax=args.kmax,
        kappamax=args.kappamax,
    )


# ---------------------------------------------------------------------------
# Stan-matrix construction and pre-computed M/N loading (moved from
# src/models/inference.py to keep inference dataset-agnostic).
# ---------------------------------------------------------------------------

import re as _re
import yaml as _yaml


def _binomial_level_probs(total_exposures, q):
    n = max(0, int(total_exposures))
    probs = np.zeros(n + 1, dtype=float)
    if n == 0 or q <= 0.0:
        probs[0] = 1.0
        return probs
    if q >= 1.0:
        probs[n] = 1.0
        return probs

    # Stable forward recursion for Binomial(n, q) PMF.
    probs[0] = (1.0 - q) ** n
    ratio = q / (1.0 - q)
    for k in range(n):
        probs[k + 1] = probs[k] * ((n - k) / float(k + 1)) * ratio

    total = probs.sum()
    if total > 0:
        probs /= total
    else:
        probs[0] = 1.0
    return probs


def _kappa_level_probs(A, q):
    counts = np.asarray(A, dtype=float)
    counts = counts[counts > 0]
    n = len(counts)

    probs = np.array([1.0], dtype=float)
    if n == 0 or q <= 0.0:
        return probs
    if q >= 1.0:
        out = np.zeros(n + 1, dtype=float)
        out[n] = 1.0
        return out

    success_probs = 1.0 - np.power(1.0 - q, counts)
    for p in success_probs:
        probs = np.convolve(probs, np.array([1.0 - p, p], dtype=float))

    total = probs.sum()
    if total > 0:
        probs /= total
    return probs


def _level_probabilities_twitter(A, q, metric):
    if metric == "k":
        return _binomial_level_probs(np.sum(np.asarray(A, dtype=float)), q)
    if metric == "kappa":
        return _kappa_level_probs(A, q)
    raise ValueError(f"Unsupported metric '{metric}'. Expected 'k' or 'kappa'.")


def _mn_row_from_level_probs(level_probs, adopted, kmax, mn_mode):
    M_vals = np.zeros(kmax, dtype=float)
    N_vals = np.zeros(kmax, dtype=float)
    if kmax <= 0:
        return M_vals, N_vals

    probs = np.asarray(level_probs, dtype=float).reshape(-1)
    if probs.size == 0:
        return M_vals, N_vals

    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if total <= 0:
        return M_vals, N_vals
    probs /= total

    max_level = probs.size - 1
    if max_level <= 0:
        return M_vals, N_vals

    mn_mode = str(mn_mode).lower()
    if mn_mode == "incidence":
        up = min(max_level, kmax)
        if adopted:
            M_vals[:up] = probs[1 : up + 1]
        else:
            N_vals[:up] = probs[1 : up + 1]
        # Event mass beyond kmax is right-censored into the terminal non-event bin.
        if max_level > kmax:
            N_vals[kmax - 1] += float(np.sum(probs[kmax + 1 :]))
        M_vals = np.clip(M_vals, 0.0, None)
        N_vals = np.clip(N_vals, 0.0, None)
        return M_vals, N_vals

    if mn_mode != "survival":
        raise ValueError(f"mn_mode must be 'incidence' or 'survival'. Got: {mn_mode}")

    cdf = np.cumsum(probs)
    nu = np.arange(1, kmax + 1, dtype=int)

    ge_mask = nu <= max_level
    gt_mask = nu < max_level

    # P(L >= nu)
    ge_vals = np.zeros(kmax, dtype=float)
    ge_vals[ge_mask] = 1.0 - cdf[nu[ge_mask] - 1]

    # P(L > nu)
    gt_vals = np.zeros(kmax, dtype=float)
    gt_vals[gt_mask] = 1.0 - cdf[nu[gt_mask]]

    if adopted:
        up = min(max_level, kmax)
        M_vals[:up] = probs[1 : up + 1]
        N_vals = gt_vals
    else:
        N_vals = ge_vals
    M_vals = np.clip(M_vals, 0.0, None)
    N_vals = np.clip(N_vals, 0.0, None)
    return M_vals, N_vals


def generate_stan_matrices_twitter(summary_df, q, metric, kmax, mn_mode="incidence"):
    V = len(summary_df)
    M = np.zeros((V, kmax), dtype=float)
    N = np.zeros((V, kmax), dtype=float)
    mn_mode = str(mn_mode).lower()
    if mn_mode not in {"incidence", "survival"}:
        raise ValueError(f"mn_mode must be 'incidence' or 'survival'. Got: {mn_mode}")

    # Row-level IPW (if present) should weight the row likelihood in Stan,
    # not the per-cell M/N counts. This prevents double-weighting artifacts.
    if 'ipw' in summary_df.columns:
        row_weights = summary_df['ipw'].to_numpy(dtype=float)
    else:
        row_weights = np.ones(V, dtype=float)

    for i, row in enumerate(summary_df.itertuples()):
        level_probs = _level_probabilities_twitter(row.A_array, q, metric)
        m_row, n_row = _mn_row_from_level_probs(
            level_probs=level_probs,
            adopted=bool(row.adopted),
            kmax=kmax,
            mn_mode=mn_mode,
        )
        M[i, :] = m_row
        N[i, :] = n_row

    return M, N, row_weights


def _sorted_exposure_cols(columns):
    cols = [c for c in columns if c != "user_id"]
    try:
        return sorted(cols, key=lambda c: int(c))
    except ValueError:
        return sorted(cols)


def _parse_q_from_mn_filename(path, kind, metric):
    pattern = _re.compile(rf"^{kind}_{_re.escape(metric)}_q(.+)\.parquet$")
    match = pattern.match(Path(path).name)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _find_q_matched_mn_file(directory, kind, metric, q, atol=1e-9):
    directory = Path(directory)
    if not directory.exists():
        return None

    target_q = float(q)
    matches = []
    for path in sorted(directory.glob(f"{kind}_{metric}_q*.parquet")):
        q_val = _parse_q_from_mn_filename(path, kind=kind, metric=metric)
        if q_val is None:
            continue
        if abs(q_val - target_q) <= atol:
            matches.append(path)

    if not matches:
        return None
    if len(matches) > 1:
        print(
            f"Warning: multiple {kind}_{metric} parquet files matched q={target_q} in {directory}. "
            f"Using {matches[0].name}."
        )
    return matches[0]


def _read_twitter_mn_parquet(path):
    df = pd.read_parquet(path).copy()
    if "user_id" not in df.columns:
        df.insert(0, "user_id", pd.Series(np.arange(len(df)), dtype=str))

    value_cols = _sorted_exposure_cols(df.columns)
    if not value_cols:
        raise ValueError(f"{path} has no exposure columns.")

    for col in value_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)

    df["user_id"] = df["user_id"].astype(str)
    df = df[["user_id"] + value_cols]
    # Be robust to accidental duplicates in upstream artifacts.
    df = df.groupby("user_id", as_index=False, sort=False)[value_cols].sum()
    return df


def _load_twitter_summary_df(summary_path):
    summary_path = Path(summary_path)
    if summary_path.is_dir():
        summary_file = summary_path / "summary.parquet"
        if not summary_file.exists():
            return None
        return pd.read_parquet(summary_file)
    return pd.read_parquet(summary_path)


def _aggregate_twitter_mn_files(paths):
    agg = None
    for path in paths:
        df = _read_twitter_mn_parquet(path).set_index("user_id")
        agg = df if agg is None else agg.add(df, fill_value=0.0)

    if agg is None:
        return None

    agg = agg.fillna(0.0)
    agg.columns = _sorted_exposure_cols(["user_id"] + list(agg.columns))
    return agg


def _resolve_twitter_processed_dir(config, summary_path):
    candidates = []
    twitter_cfg = config.get("twitter", {}) if isinstance(config.get("twitter", {}), dict) else {}

    # Highest-priority explicit overrides.
    for key in ("twitter_processed_dir", "processed_dir"):
        val = config.get(key)
        if val:
            candidates.append(Path(val))
    val = twitter_cfg.get("processed_dir")
    if val:
        candidates.append(Path(val))

    # Data-prep config referenced by the sweep config.
    data_config_path = config.get("data_config")
    if data_config_path:
        cfg_path = Path(data_config_path)
        if not cfg_path.is_absolute():
            cfg_path = Path.cwd() / cfg_path
        if cfg_path.exists():
            try:
                with open(cfg_path, "r") as f:
                    prep_cfg = _yaml.safe_load(f) or {}
                prep_dir = prep_cfg.get("processed_dir")
                if prep_dir:
                    candidates.append(Path(prep_dir))
            except Exception as e:
                print(f"Warning: could not load data_config '{cfg_path}': {e}")

    # Conservative fallbacks.
    summary_name = Path(summary_path).stem.lower()
    if "survival" in summary_name:
        candidates.append(Path("data/processed/twitter_survival"))
    candidates.append(Path("data/processed/twitter"))

    for candidate in candidates:
        candidate = Path(candidate)
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        if candidate.exists():
            return candidate
    return None


def _load_precomputed_twitter_mn(summary_df, config, summary_path, q, metric):
    summary_df = summary_df.copy() if summary_df is not None else pd.DataFrame()
    if not summary_df.empty and "user_id" in summary_df.columns and summary_df["user_id"].astype(str).duplicated().any():
        # Without intervention_id in summary rows, duplicate user_id rows are ambiguous.
        print(
            "Warning: summary has duplicate user_id rows; cannot align pre-computed "
            "Twitter M/N uniquely. Falling back to A_array reconstruction."
        )
        return None

    summary_path = Path(summary_path)
    m_indexed = None
    n_indexed = None
    source_desc = None

    if summary_path.is_dir():
        direct_m = summary_path / "M.parquet"
        direct_n = summary_path / "N.parquet"
        if direct_m.exists() and direct_n.exists():
            m_indexed = _read_twitter_mn_parquet(direct_m).set_index("user_id")
            n_indexed = _read_twitter_mn_parquet(direct_n).set_index("user_id")
            source_desc = f"{summary_path} (direct)"

    if m_indexed is None or n_indexed is None:
        processed_dir = _resolve_twitter_processed_dir(config, summary_path)
        if processed_dir is None:
            return None

        full_dir = processed_dir / "full"
        m_full = _find_q_matched_mn_file(full_dir, kind="M", metric=metric, q=q)
        n_full = _find_q_matched_mn_file(full_dir, kind="N", metric=metric, q=q)
        if m_full is not None and n_full is not None:
            m_indexed = _read_twitter_mn_parquet(m_full).set_index("user_id")
            n_indexed = _read_twitter_mn_parquet(n_full).set_index("user_id")
            source_desc = f"{processed_dir}/full"
        else:
            m_files = []
            n_files = []
            for d in sorted(processed_dir.iterdir()):
                if not d.is_dir() or d.name in {"full", "summaries"}:
                    continue
                m_path = _find_q_matched_mn_file(d, kind="M", metric=metric, q=q)
                n_path = _find_q_matched_mn_file(d, kind="N", metric=metric, q=q)
                if m_path is not None and n_path is not None:
                    m_files.append(m_path)
                    n_files.append(n_path)

            if m_files and n_files:
                m_indexed = _aggregate_twitter_mn_files(m_files)
                n_indexed = _aggregate_twitter_mn_files(n_files)
                source_desc = f"{processed_dir}/* (aggregated)"

    if m_indexed is None or n_indexed is None:
        return None

    # Ensure aligned exposure columns.
    exposure_cols = _sorted_exposure_cols(["user_id"] + list(set(m_indexed.columns) | set(n_indexed.columns)))
    m_indexed = m_indexed.reindex(columns=exposure_cols, fill_value=0.0)
    n_indexed = n_indexed.reindex(columns=exposure_cols, fill_value=0.0)

    if summary_df.empty or "user_id" not in summary_df.columns:
        selected_ids = list(m_indexed.index)
    else:
        selected_ids = summary_df["user_id"].astype(str).tolist()
        missing = sorted(set(selected_ids) - set(m_indexed.index))
        if missing:
            # Summary and processed artifacts are out of sync; fall back to reconstruction.
            print(
                f"Warning: precomputed Twitter M/N missing {len(missing)} users "
                f"(example: {missing[:3]}). Falling back to summary reconstruction."
            )
            return None

    M = m_indexed.loc[selected_ids, exposure_cols].to_numpy(dtype=float)
    N = n_indexed.loc[selected_ids, exposure_cols].to_numpy(dtype=float)
    return M, N, source_desc
