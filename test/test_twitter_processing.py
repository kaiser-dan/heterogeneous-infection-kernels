from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src.data_processing.process_twitter import (
    _mn_rows_from_prob_dist,
    aggregate_all_interventions,
    process_intervention,
    save_matrices,
)


def _twitter_timestamp(day: int) -> str:
    return f"Wed Jan {day:02d} 00:00:00 +0000 2020"


def _tweet(tweet_id: int, user_id: int, day: int, *, retweeted_status=None):
    tweet = {
        "id": tweet_id,
        "user": {"id": user_id},
        "created_at": _twitter_timestamp(day),
        "entities": {"hashtags": []},
    }
    if retweeted_status is not None:
        tweet["retweeted_status"] = retweeted_status
    return tweet


def test_process_intervention_skips_post_adoption_exposures():
    graph = nx.Graph()
    graph.add_node(1, account_type="bot")
    graph.add_node(10, account_type="user")
    graph.add_edge(1, 10)

    intervention = "campaign"
    intervention_tweets = {
        intervention: [
            _tweet(100, 1, 1),
            _tweet(101, 1, 3),
        ]
    }
    intervention_user_retweets = {intervention: [_tweet(500, 10, 2)]}
    intervention_bot_retweets = {intervention: []}

    user_data = process_intervention(
        intervention,
        graph,
        intervention_tweets,
        intervention_user_retweets,
        intervention_bot_retweets,
        bot_IDs=set(),
        q_sc=1.0,
        q_cc=1.0,
    )

    assert len(user_data) == 1
    record = user_data[0]
    assert record["user_id"] == "10"
    assert record["adopted"] is True
    assert record["bot_exposures"] == {1: 1}
    assert record["p_dist_k"].get(1, 0.0) == pytest.approx(1.0)
    assert record["p_dist_kappa"].get(1, 0.0) == pytest.approx(1.0)


def test_process_intervention_retweets_can_use_origin_or_proxy_bot():
    graph = nx.Graph()
    graph.add_node(1, account_type="bot")
    graph.add_node(2, account_type="bot")
    graph.add_node(10, account_type="user")
    graph.add_edge(2, 10)

    intervention = "campaign"
    origin = {"id": 100, "user": {"id": 1}}
    intervention_tweets = {intervention: []}
    intervention_user_retweets = {intervention: []}
    intervention_bot_retweets = {intervention: [_tweet(200, 2, 1, retweeted_status=origin)]}

    origin_counts = process_intervention(
        intervention,
        graph,
        intervention_tweets,
        intervention_user_retweets,
        intervention_bot_retweets,
        bot_IDs=set(),
        q_sc=1.0,
        q_cc=1.0,
        count_retweets_as_exposures=True,
    )
    proxy_counts = process_intervention(
        intervention,
        graph,
        intervention_tweets,
        intervention_user_retweets,
        intervention_bot_retweets,
        bot_IDs=set(),
        q_sc=1.0,
        q_cc=1.0,
        count_retweets_as_exposures=False,
    )

    assert origin_counts[0]["bot_exposures"] == {1: 1}
    assert proxy_counts[0]["bot_exposures"] == {2: 1}


def test_mn_rows_from_prob_dist_survival_tracks_remaining_risk():
    m_vals, n_vals = _mn_rows_from_prob_dist(
        {1: 0.25, 2: 0.75},
        adopted=True,
        kmax=3,
        mn_mode="survival",
    )
    assert np.allclose(m_vals, [0.25, 0.75, 0.0])
    assert np.allclose(n_vals, [0.75, 0.0, 0.0])


def test_aggregate_all_interventions_sums_saved_rows(tmp_path):
    user_data_iv1 = [
        {
            "user_id": "u1",
            "p_dist_k": {1: 1.0},
            "p_dist_kappa": {1: 1.0},
            "adopted": True,
            "bot_exposures": {1: 1},
        },
        {
            "user_id": "u2",
            "p_dist_k": {2: 1.0},
            "p_dist_kappa": {2: 1.0},
            "adopted": False,
            "bot_exposures": {1: 2},
        },
    ]
    user_data_iv2 = [
        {
            "user_id": "u1",
            "p_dist_k": {2: 1.0},
            "p_dist_kappa": {1: 1.0},
            "adopted": False,
            "bot_exposures": {2: 1},
        },
        {
            "user_id": "u3",
            "p_dist_k": {1: 1.0},
            "p_dist_kappa": {1: 1.0},
            "adopted": True,
            "bot_exposures": {3: 1},
        },
    ]

    save_matrices(user_data_iv1, "iv1", processed_dir=tmp_path, q_sc=1.0, q_cc=1.0, kmax=3, kappamax=3)
    save_matrices(user_data_iv2, "iv2", processed_dir=tmp_path, q_sc=1.0, q_cc=1.0, kmax=3, kappamax=3)
    aggregate_all_interventions(processed_dir=tmp_path, q_sc=1.0, q_cc=1.0)

    full_m = pd.read_parquet(Path(tmp_path) / "full" / "M_k_q1.0.parquet").set_index("user_id")
    full_n = pd.read_parquet(Path(tmp_path) / "full" / "N_k_q1.0.parquet").set_index("user_id")

    assert full_m.loc["u1", "1"] == pytest.approx(1.0)
    assert full_n.loc["u1", "2"] == pytest.approx(1.0)
    assert full_n.loc["u2", "2"] == pytest.approx(1.0)
    assert full_m.loc["u3", "1"] == pytest.approx(1.0)
