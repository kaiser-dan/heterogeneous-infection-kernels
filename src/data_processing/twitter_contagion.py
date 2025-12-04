from pathlib import Path
import datetime
import json
import sqlite3

import networkx as nx
import numpy as np
import pandas as pd

__all__ = [
    "follow_adjacency",
    "retweet_time_series",
    "get_cascade_bot",
    "get_cascade_hashtag",
    "filter_cascades_by_intervention",
    "compute_exposure_counts",
    "make_graph",
    "make_graph_alternate",
    "load_bot_tweets",
    "load_retweet_cascades",
    "get_id_to_bot_tweet",
    "get_id_to_retweets",
    "get_bot_ids",
    "get_interventions",
]


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "twitter_contagion_data"
DEFAULT_BOT_TWEETS = "all_bot_tweets_"
DEFAULT_RETWEETS = "retweets_"
DEFAULT_EDGES = "edges"
DEFAULT_TARGETS_DB = "targets.db"

BOT_ID_TO_HANDLE = {
    2749655899: "FitVeganGirl_",
    2786086363: "AndrenatorC",
    2786246510: "Van_Jester",
    2786363222: "SimonWJorgensen",
    2786427865: "SuperRexy",
    2787456828: "matsie_at_dtu",
    2787463746: "TheRexyGuy",
    2787465462: "RealAndersDuck",
    2787474126: "zakflanigan",
    2787481304: "Pralesworth",
    2787482192: "marcussor",
    2787485774: "ioapsy",
    2787485978: "Awe5omeMike",
    2787486036: "AnnasHollywood",
    2787486060: "sonia_manning",
    2787489548: "ethanwoods88",
    2787498932: "LakerolIs",
    2787499142: "madpopo79",
    2787499932: "EmiiiSuu",
    2787500612: "AxelCyrilian",
    2787503576: "THINKDEEPYO",
    2787514615: "Shtinoehh",
    2787516307: "SirZenji",
    2787527221: "RichardsIndie",
    2787531548: "CPH_Startup",
    2787540506: "canuckWong",
    2787554990: "SusanooGirl",
    2787610574: "AlyciaGerald",
    2787613616: "MyNikkles",
    2787620990: "meetjamesmet",
    2787739466: "tfosterm",
    2787833887: "ClintCrock",
    2787905696: "JackBoHorseMan",
    2788164512: "hirihiker",
    2795790799: "hybrishybris",
    2801335412: "jsmth_t",
    2802511814: "neergdave",
    2829847496: "ericfullhammer",
    2841459034: "zoesprings",
}

INTERVENTION_HASHTAGS = [
    "getyourflushot",
    "highfiveastranger",
    "somethinggood",
    "HowManyPushups",
    "turkeyface",
    "SFThanks",
    "blackfridaystories",
    "BanksySF",
]


def _data_path(filename, data_dir=None):
    """Resolve a dataset-relative path."""
    if filename is None:
        raise ValueError("filename must be provided")
    base = Path(data_dir) if data_dir is not None else DATA_DIR
    candidate = Path(filename)
    if candidate.is_absolute():
        return candidate
    return base / candidate


def _is_missing(value):
    """Return True if value represents a missing entry."""
    try:
        return pd.isna(value)
    except TypeError:
        return value is None


def _coerce_int_id(value):
    """Convert a value into an integer ID if possible."""
    if _is_missing(value):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        return int(value)
    value_str = str(value).strip()
    if not value_str:
        return None
    if value_str.startswith("+"):
        value_str = value_str[1:]
    if value_str.isdigit():
        return int(value_str)
    if value_str.startswith("-") and value_str[1:].isdigit():
        return int(value_str)
    try:
        return int(value_str)
    except ValueError:
        try:
            return int(float(value_str))
        except (TypeError, ValueError):
            return None


def _coerce_timestamp(value):
    """Convert a database timestamp string into a pandas Timestamp."""
    if _is_missing(value):
        return None
    value_str = str(value).strip()
    if not value_str:
        return None
    try:
        return _parse_db_timestamp(value_str)
    except ValueError:
        return None


def _min_timestamp(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def _max_timestamp(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def _load_target_handles(targets_path):
    """Load mapping from account id to handle from the SQLite DB."""
    path = Path(targets_path)
    if not path.exists():
        raise FileNotFoundError(f"targets database not found: {path}")
    handles = {}
    conn = sqlite3.connect(str(path))
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id_str, handle FROM targets")
        for id_str, handle in cursor:
            node_id = _coerce_int_id(id_str)
            if node_id is None:
                continue
            handles[node_id] = handle
    finally:
        conn.close()
    return handles


def _annotate_node_metadata(graph, user_handles, include_details):
    for node in graph.nodes:
        if node in user_handles:
            graph.nodes[node].setdefault("handle", user_handles[node])
        elif node in BOT_ID_TO_HANDLE:
            graph.nodes[node].setdefault("handle", BOT_ID_TO_HANDLE[node])
        if include_details:
            graph.nodes[node]["account_type"] = (
                "bot" if node in BOT_ID_TO_HANDLE else "user"
            )


def load_bot_tweets(filename=DEFAULT_BOT_TWEETS, *, data_dir=None):
    """
    Load the ``all_bot_tweets_`` file.
    """
    path = _data_path(filename, data_dir)
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_retweet_cascades(filename=DEFAULT_RETWEETS, *, data_dir=None):
    """
    Load the ``retweets_`` file which contains all cascades.
    """
    path = _data_path(filename, data_dir)
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def get_id_to_bot_tweet(bot_tweets=None, *, filename=DEFAULT_BOT_TWEETS, data_dir=None):
    """
    Map original bot tweet IDs to the tweet dictionaries.
    """
    tweets = bot_tweets if bot_tweets is not None else load_bot_tweets(
        filename=filename, data_dir=data_dir
    )
    return {
        tweet["id"]: tweet
        for tweet in tweets
        if "retweeted_status" not in tweet
    }


def get_id_to_retweets(
    retweets=None, *, filename=DEFAULT_RETWEETS, data_dir=None, sort=True
):
    """
    Map original bot tweet IDs to their retweet cascades.
    """
    cascades = retweets if retweets is not None else load_retweet_cascades(
        filename=filename, data_dir=data_dir
    )
    id_to_retweets = {}
    for cascade in cascades:
        if not cascade:
            continue
        status = cascade[0].get("retweeted_status")
        if not status:
            continue
        original_id = status["id"]
        tweets = list(cascade)
        if sort:
            tweets.sort(key=lambda t: _parse_twitter_timestamp(t["created_at"]))
        id_to_retweets[original_id] = tweets
    return id_to_retweets


def get_bot_ids():
    return set(BOT_ID_TO_HANDLE.keys())


def get_interventions():
    return list(INTERVENTION_HASHTAGS)


def build_adjacency_for_participants(
    edges,
    participants,
    *,
    include_bots=True,
    directed=True,
    relation_type=None,
    bot_column="bot_id",
    user_column="account_id",
    relation_column="relation",
):
    """
    Construct a follow-adjacency matrix restricted to a set of participants.

    Parameters
    ----------
    edges : pandas.DataFrame
        The full edges table.
    participants : Iterable[int]
        Set of account ids to keep (users + bots that appear in a cascade).
    include_bots : bool
        Whether bot columns appear in the adjacency (defaults to True).
    directed : bool
        Forwarded to ``follow_adjacency``.
    relation_type : str or None
        Optional relation filter (see ``follow_adjacency``).
    """
    if edges is None or edges.empty:
        raise ValueError("edges DataFrame must be provided")
    if not participants:
        raise ValueError("participants must contain at least one node")

    participants = set(participants)
    filtered = edges[edges[user_column].isin(participants)].copy()
    if filtered.empty:
        raise ValueError("no edges connect the requested participants")

    adjacency = follow_adjacency(
        filtered,
        directed=directed,
        relation_type=relation_type,
        bot_column=bot_column,
        user_column=user_column,
        relation_column=relation_column,
        include_bots=include_bots,
    )

    keep_nodes = [node for node in adjacency.index if node in participants]
    if keep_nodes:
        adjacency = adjacency.loc[keep_nodes, :]
        adjacency = adjacency.loc[:, keep_nodes]
    return adjacency


def make_graph(
    include_details=True,
    *,
    data_dir=None,
    edges_file=DEFAULT_EDGES,
    targets_db=DEFAULT_TARGETS_DB,
):
    """
    Build a graph of bots and users directly from ``targets.db`` and ``edges``.

    Set ``include_details=False`` to ignore unreliable ``first_seen`` timestamps.
    """
    edges_path = _data_path(edges_file, data_dir)
    targets_path = _data_path(targets_db, data_dir)
    user_handles = _load_target_handles(targets_path)
    edges = pd.read_csv(edges_path, low_memory=False)

    G = nx.Graph()

    for _, row in edges.iterrows():
        account_id = _coerce_int_id(row.get("account_id"))
        bot_id = _coerce_int_id(row.get("bot_id"))
        if account_id is None or bot_id is None:
            continue

        relation = str(row.get("relation") or "").strip()

        if include_details:
            first_seen = _coerce_timestamp(row.get("first_seen"))
            last_seen = _coerce_timestamp(row.get("last_seen"))
            if G.has_edge(account_id, bot_id):
                edge = G[account_id][bot_id]
                edge["first_seen"] = _min_timestamp(edge.get("first_seen"), first_seen)
                edge["last_seen"] = _max_timestamp(edge.get("last_seen"), last_seen)
                prev_relation = edge.get("relation")
                if prev_relation == "follower" or relation == "follower":
                    edge["relation"] = "follower"
                else:
                    edge["relation"] = prev_relation or relation
            else:
                G.add_edge(
                    account_id,
                    bot_id,
                    relation=relation,
                    first_seen=first_seen,
                    last_seen=last_seen,
                )
        else:
            G.add_edge(account_id, bot_id)

    _annotate_node_metadata(G, user_handles, include_details)
    return G


def make_graph_alternate(
    include_details=True,
    *,
    data_dir=None,
    edges_file=DEFAULT_EDGES,
    targets_db=DEFAULT_TARGETS_DB,
):
    """
    Alternate graph builder that stores all ``first_seen``/``last_seen`` intervals.
    """
    edges_path = _data_path(edges_file, data_dir)
    targets_path = _data_path(targets_db, data_dir)
    user_handles = _load_target_handles(targets_path)
    edges = pd.read_csv(edges_path, low_memory=False)

    G = nx.Graph()

    for _, row in edges.iterrows():
        account_id = _coerce_int_id(row.get("account_id"))
        bot_id = _coerce_int_id(row.get("bot_id"))
        if account_id is None or bot_id is None:
            continue

        relation = str(row.get("relation") or "").strip()
        interval = (
            _coerce_timestamp(row.get("first_seen")),
            _coerce_timestamp(row.get("last_seen")),
        )

        if G.has_edge(account_id, bot_id):
            G[account_id][bot_id].setdefault("seen", []).append(interval)
        else:
            G.add_edge(
                account_id,
                bot_id,
                relation=relation,
                seen=[interval],
            )

    _annotate_node_metadata(G, user_handles, include_details)
    return G


def _parse_twitter_timestamp(created_at):
    dt = datetime.datetime.strptime(created_at, '%a %b %d %H:%M:%S +0000 %Y')
    return pd.Timestamp(dt, tz='UTC')
def _parse_db_timestamp(timestamp_str):
    try:
        dt = datetime.datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
    except ValueError:
        dt = datetime.datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
    return pd.Timestamp(dt, tz='UTC')
def _to_timestamp(value):
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.tz_convert("UTC") if value.tzinfo else value.tz_localize("UTC")
    return pd.to_datetime(value, utc=True)


def follow_adjacency(
    edges,
    *,
    directed=True,
    relation_type=None,
    bot_column="bot_id",
    user_column="account_id",
    relation_column="relation",
    include_bots=True,
):
    """
    Build an adjacency matrix for Twitter follow relationships.

    Parameters
    ----------
    edges : pandas.DataFrame
        DataFrame containing Twitter follow relationships (see ``edges`` file).
    directed : bool
        Whether to preserve the direction of follow relationships.
    relation_type : str or None
        Filter by relation type: "follower" (user follows bot), "friend"
    bot_column : str
        Column name for bot IDs.
    user_column : str
        Column name for user IDs.
    relation_column : str
        Column name for relation type.
    include_bots : bool
        Whether to include bots as nodes in the adjacency matrix. If False,
        only creates a user-user adjacency based on shared bot follows.
        Defaults to True.

    Returns
    -------
    pandas.DataFrame
        Square adjacency matrix indexed/columned by account IDs. For directed
        graphs, adjacency[i,j] = 1 means account i follows account j.
    """
    # Filter by relation type if specified
    if relation_type is not None:
        edges = edges[edges[relation_column] == relation_type].copy()
    else:
        edges = edges.copy()

    # Drop rows with missing IDs
    edges = edges.dropna(subset=[bot_column, user_column])

    # Convert IDs to integers, dropping rows that can't be converted
    # (some test accounts have string IDs like "radus")
    try:
        edges[bot_column] = pd.to_numeric(edges[bot_column], errors='coerce')
        edges[user_column] = pd.to_numeric(edges[user_column], errors='coerce')
        edges = edges.dropna(subset=[bot_column, user_column])
        edges[bot_column] = edges[bot_column].astype(int)
        edges[user_column] = edges[user_column].astype(int)
    except Exception:
        # If conversion fails, filter manually
        edges = edges[edges[bot_column].apply(lambda x: str(x).isdigit())]
        edges = edges[edges[user_column].apply(lambda x: str(x).isdigit())]
        edges[bot_column] = edges[bot_column].astype(int)
        edges[user_column] = edges[user_column].astype(int)

    # Get all unique nodes
    if include_bots:
        all_nodes = pd.concat([edges[bot_column], edges[user_column]],
                              ignore_index=True).unique()
    else:
        # Only include users
        all_nodes = edges[user_column].unique()

    nodes = pd.Index(sorted(all_nodes))

    # Initialize adjacency matrix
    adjacency = pd.DataFrame(
        data=np.zeros((len(nodes), len(nodes)), dtype=int),
        index=nodes,
        columns=nodes
    )

    for _, row in edges.iterrows():
        user_id = row[user_column]
        bot_id = row[bot_column]
        relation = row[relation_column]

        # Skip if nodes aren't in our matrix (e.g., if not include_bots)
        if user_id not in adjacency.index or bot_id not in adjacency.columns:
            continue

        if relation == "follower":
            # user follows bot
            adjacency.loc[user_id, bot_id] = 1
            if not directed:
                adjacency.loc[bot_id, user_id] = 1
        elif relation == "friend":
            # bot follows user
            adjacency.loc[bot_id, user_id] = 1
            if not directed:
                adjacency.loc[user_id, bot_id] = 1

    return adjacency


def retweet_time_series(
    retweets,
    cascade_id,
    *,
    freq="6H",
    participants=None,
    start_time=None,
    end_time=None,
    include_source=True,
    bot_tweets=None,
):
    """
    Create a discretized retweet indicator matrix for a cascade.

    Parameters
    ----------
    retweets : list of lists
        List of cascades, where each cascade is a list of retweet dictionaries.
    cascade_id : int
        Index of the cascade to analyze (0 to len(retweets)-1).
    freq : str
        Pandas offset alias defining the time resolution ("1H", "1D", etc.).
        Defaults to "1H".
    participants : iterable or None
        Optional iterable of user IDs to include even if they didn't retweet.
    start_time : various or None
        Optional override for the time axis start. Can be epoch seconds,
        ISO timestamp string, or pandas.Timestamp object.
    end_time : various or None
        Optional override for the time axis end.
    include_source : bool
        Whether to mark the source bot as having "retweeted" at time of
        original tweet. Defaults to True.
    bot_tweets : list or None
        Optional list of bot tweet dictionaries (from all_bot_tweets_ file).
        Required if include_source=True to identify the source bot and time.

    Returns
    -------
    pandas.DataFrame
        Rows represent discretized time bins and columns represent user IDs.
        Values are 0 prior to retweet and 1 afterwards.
    """
    # Get the cascade
    if cascade_id < 0 or cascade_id >= len(retweets):
        raise ValueError(f"cascade_id {cascade_id} out of range [0, {len(retweets)-1}]")

    cascade = retweets[cascade_id]

    if len(cascade) == 0:
        raise ValueError(f"Cascade {cascade_id} has no retweets")

    # Extract retweet events (user_id, timestamp)
    retweet_events = []
    original_tweet_id = None

    for retweet in cascade:
        user_id = retweet['user']['id']
        timestamp = _parse_twitter_timestamp(retweet['created_at'])
        retweet_events.append((user_id, timestamp))

        # Get original tweet ID
        if 'retweeted_status' in retweet and original_tweet_id is None:
            original_tweet_id = retweet['retweeted_status']['id']

    # Convert to DataFrame for easier manipulation
    retweet_df = pd.DataFrame(retweet_events, columns=['user_id', 'timestamp'])

    # Determine time range
    start_ts = _to_timestamp(start_time) or retweet_df['timestamp'].min()
    end_ts = _to_timestamp(end_time) or retweet_df['timestamp'].max()

    # If include_source, find the original tweet and add the bot
    if include_source and bot_tweets is not None and original_tweet_id is not None:
        # Find the original tweet in bot_tweets
        for tweet in bot_tweets:
            if tweet['id'] == original_tweet_id:
                bot_id = tweet['user']['id']
                bot_timestamp = _parse_twitter_timestamp(tweet['created_at'])

                # Add bot to retweet events at original tweet time
                retweet_df = pd.concat([
                    pd.DataFrame([{'user_id': bot_id, 'timestamp': bot_timestamp}]),
                    retweet_df
                ], ignore_index=True)

                # Update start time if needed
                start_ts = min(start_ts, bot_timestamp)
                break

    # Get all users
    user_ids = set(retweet_df['user_id'].unique())
    if participants is not None:
        user_ids.update(int(uid) for uid in participants)
    user_index = sorted(user_ids)

    # Create timeline (convert freq to lowercase for pandas 2.0+ compatibility)
    timeline = pd.date_range(start=start_ts, end=end_ts, freq=freq.lower())

    # Initialize states matrix
    states = pd.DataFrame(
        data=0,
        index=timeline,
        columns=user_index,
        dtype=int
    )

    # Fill in retweet events
    for _, event in retweet_df.iterrows():
        uid = event['user_id']
        retweet_ts = event['timestamp']

        # Mark as 1 from the retweet time onwards
        states.loc[states.index >= retweet_ts, uid] = 1

    return states


def get_cascade_bot(retweets, cascade_id, bot_tweets):
    """
    Get the bot ID that posted the original tweet for a cascade.

    Parameters
    ----------
    retweets : list of lists
        List of cascades.
    cascade_id : int
        Index of the cascade.
    bot_tweets : list
        List of bot tweet dictionaries.

    Returns
    -------
    int or None
        Bot user ID, or None if not found.
    """
    if cascade_id < 0 or cascade_id >= len(retweets):
        return None

    cascade = retweets[cascade_id]
    if len(cascade) == 0:
        return None

    # Get original tweet ID from first retweet
    original_tweet_id = None
    for retweet in cascade:
        if 'retweeted_status' in retweet:
            original_tweet_id = retweet['retweeted_status']['id']
            break

    if original_tweet_id is None:
        return None

    # Find the bot who posted it
    for tweet in bot_tweets:
        if tweet['id'] == original_tweet_id:
            return tweet['user']['id']

    return None


def get_cascade_hashtag(retweets, cascade_id, bot_tweets):
    """
    Get the primary hashtag (intervention) for a cascade.

    Parameters
    ----------
    retweets : list of lists
        List of cascades.
    cascade_id : int
        Index of the cascade.
    bot_tweets : list
        List of bot tweet dictionaries.

    Returns
    -------
    str or None
        Primary hashtag (lowercase), or None if no hashtags found.
    """
    if cascade_id < 0 or cascade_id >= len(retweets):
        return None

    cascade = retweets[cascade_id]
    if len(cascade) == 0:
        return None

    # Get original tweet ID
    original_tweet_id = None
    for retweet in cascade:
        if 'retweeted_status' in retweet:
            original_tweet_id = retweet['retweeted_status']['id']
            break

    if original_tweet_id is None:
        return None

    # Find the original tweet and extract hashtag
    for tweet in bot_tweets:
        if tweet['id'] == original_tweet_id:
            hashtags = tweet.get('entities', {}).get('hashtags', [])
            if hashtags:
                # Return first hashtag (lowercase)
                return hashtags[0]['text'].lower()
            return None

    return None


def filter_cascades_by_intervention(retweets, bot_tweets, intervention):
    """
    Get list of cascade IDs for a specific intervention hashtag.

    Parameters
    ----------
    retweets : list of lists
        List of cascades.
    bot_tweets : list
        List of bot tweet dictionaries.
    intervention : str
        Intervention hashtag to filter by (case-insensitive).

    Returns
    -------
    list of int
        List of cascade IDs that match the intervention.
    """
    intervention = intervention.lower()
    matching_cascades = []

    for cascade_id in range(len(retweets)):
        hashtag = get_cascade_hashtag(retweets, cascade_id, bot_tweets)
        if hashtag and hashtag == intervention:
            matching_cascades.append(cascade_id)

    return matching_cascades


def compute_exposure_counts(
    retweets,
    cascade_id,
    adjacency,
    *,
    bot_tweets=None,
    freq="1H",
    include_source=True,
    participants=None,
    kmax=6,
):
    """
    Build M and N exposure count matrices for a set of cascades.

    Parameters
    ----------
    retweets : list
        Output of the ``retweets_`` JSON file (list of cascades).
    cascade_id : int
        Index of the cascade to analyze.
    adjacency : pandas.DataFrame
        Square matrix produced by ``follow_adjacency``. ``adjacency[i,j]=1``
        means node i follows node j (j can expose i).
    bot_tweets : list or None
        List of original bot tweets. Required when ``include_source`` is True.
    freq : str
        Time discretization passed to ``retweet_time_series``.
    include_source : bool
        Whether to include the original bot in the time series.
    participants : iterable or None
        Optional whitelist of nodes to include in the time series. Nodes not
        present in ``adjacency`` are ignored.
    kmax : int
        Maximum neighbor count bucket. Exposure counts >= kmax-1 overflow into
        the last column.

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        Tuple ``(M, N)`` where each DataFrame has index equal to node ids and
        columns ``0..kmax-1``.
    """
    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be a square matrix")

    adjacency = adjacency.copy()
    adjacency = adjacency.fillna(0)
    adjacency = adjacency.astype(int)

    if not adjacency.index.equals(adjacency.columns):
        adjacency = adjacency.loc[adjacency.index.intersection(adjacency.columns), :]
        adjacency = adjacency.loc[:, adjacency.index]

    nodes = list(adjacency.index)
    if len(nodes) == 0:
        raise ValueError("adjacency matrix is empty")

    adj_matrix = adjacency.values
    V = len(nodes)

    if participants is None:
        participants = nodes
    else:
        participants = [node for node in participants if node in adjacency.index]
        if not participants:
            raise ValueError("participants must include at least one node in adjacency")

    if kmax < 1:
        raise ValueError("kmax must be >= 1")

    M = np.zeros((V, kmax), dtype=int)
    N = np.zeros((V, kmax), dtype=int)

    states = retweet_time_series(
        retweets,
        cascade_id,
        freq=freq,
        participants=participants,
        include_source=include_source,
        bot_tweets=bot_tweets,
    )

    states = states.reindex(columns=nodes, fill_value=0)
    state_values = states.values.astype(int, copy=False)
    exposure_values = state_values.dot(adj_matrix.T)
    adopted_mask = state_values.astype(bool, copy=False)

    for node_idx in range(V):
        exposures = exposure_values[:, node_idx]
        if adopted_mask[:, node_idx].any():
            adopt_bin = adopted_mask[:, node_idx].argmax()
            failed_exposures = exposures[:adopt_bin]
            if failed_exposures.size:
                buckets = np.minimum(failed_exposures.astype(int), kmax - 1)
                counts = np.bincount(buckets, minlength=kmax)
                N[node_idx, :kmax] += counts[:kmax]

            success_bucket = min(int(exposures[adopt_bin]), kmax - 1)
            M[node_idx, success_bucket] += 1
        else:
            if exposures.size:
                buckets = np.minimum(exposures.astype(int), kmax - 1)
                counts = np.bincount(buckets, minlength=kmax)
                N[node_idx, :kmax] += counts[:kmax]

    columns = list(range(kmax))
    M_df = pd.DataFrame(M, index=nodes, columns=columns)
    N_df = pd.DataFrame(N, index=nodes, columns=columns)
    return M_df, N_df
