"""WeChat Moments (WMdata, Zhang et al. 2018) logs -> friend graph and activation counts (M, N).

A trial is one (user, page) pair, taken at the user's first view of the page. nu is the number of
distinct friends who had shared the page before that view; the trial succeeds if the user shares it
afterwards. Users who shared before any friend did (seeds) give no trial. nu > kmax pools into kmax.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["page", "pub", "viewer", "type", "t", "js"]


def read_events(paths) -> pd.DataFrame:
    """View and share records from the raw tab-separated logs, time in seconds."""
    frames = []
    for path in paths:
        d = pd.read_csv(path, sep="\t", header=None, names=COLUMNS, quoting=3, on_bad_lines="skip",
                        dtype={"type": "string"}, usecols=range(5))
        d = d[d["type"].isin(["view", "share"])]
        d["t"] = pd.to_numeric(d["t"], errors="coerce") / 1000.0
        frames.append(d.dropna())
    ev = pd.concat(frames, ignore_index=True).sort_values("t", kind="stable")
    return ev.astype({"page": int, "pub": int, "viewer": int, "type": str})


def friend_edges(ev: pd.DataFrame) -> pd.DataFrame:
    """Undirected friendships revealed by views of a friend's share (pub > 0, pub != viewer)."""
    v = ev[(ev["type"] == "view") & (ev["pub"] > 0) & (ev["pub"] != ev["viewer"])]
    uv = np.unique(np.sort(v[["pub", "viewer"]].to_numpy(), axis=1), axis=0)
    return pd.DataFrame(uv, columns=["u", "v"])


def share_times(ev: pd.DataFrame) -> pd.DataFrame:
    """First time each user is known to have shared each page: a share record or publishing a viewed page."""
    s = ev.loc[ev["type"] == "share", ["page", "viewer", "t"]].rename(columns={"viewer": "user"})
    p = ev.loc[(ev["type"] == "view") & (ev["pub"] > 0), ["page", "pub", "t"]].rename(columns={"pub": "user"})
    return pd.concat([s, p]).groupby(["page", "user"], as_index=False)["t"].min()


def view_trials(ev: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """One row per (user, page) trial: nu and whether the user adopted."""
    both = pd.concat([edges, edges.rename(columns={"u": "v", "v": "u"})])
    st = share_times(ev)
    expo = st.rename(columns={"user": "v", "t": "t_friend"}).merge(both, on="v")
    views = ev[ev["type"] == "view"].groupby(["viewer", "page"], as_index=False)["t"].min()
    views = views.rename(columns={"viewer": "u", "t": "t_view"})

    g = expo.merge(views, on=["page", "u"])
    g = g[g["t_friend"] < g["t_view"]]
    tr = g.groupby(["u", "page"], as_index=False).agg(nu=("t_friend", "size"), t_view=("t_view", "first"))
    own = st.rename(columns={"user": "u", "t": "t_own"})
    tr = tr.merge(own, on=["page", "u"], how="left").fillna({"t_own": np.inf})
    tr = tr[tr["t_own"] >= tr["t_view"]]  # share recorded before the view
    tr["adopt"] = np.isfinite(tr["t_own"])
    return tr.rename(columns={"u": "user"})[["user", "page", "nu", "adopt"]]


def wechat_MN(paths, kmax: int = 20):
    """(M, N, users, edges) for the view-conditioned WeChat data.

    M[i, j], N[i, j] count adoptions and non-adoptions of user i at nu = j + 1.
    edges are friendships among `users`, as row indices into M.
    """
    ev = read_events([Path(p) for p in paths])
    friends = friend_edges(ev)
    tr = view_trials(ev, friends)

    users = np.sort(tr["user"].unique())
    i = np.searchsorted(users, tr["user"])
    j = np.minimum(tr["nu"].to_numpy(), kmax) - 1
    M = np.zeros((len(users), kmax))
    N = np.zeros((len(users), kmax))
    np.add.at(M, (i, j), tr["adopt"].to_numpy())
    np.add.at(N, (i, j), ~tr["adopt"].to_numpy())

    keep = friends["u"].isin(users) & friends["v"].isin(users)
    edges = np.c_[np.searchsorted(users, friends.loc[keep, "u"]), np.searchsorted(users, friends.loc[keep, "v"])]
    return M, N, users, edges
