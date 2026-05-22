"""
Generate M_kappa_q0.182 and N_kappa_q0.182 in data/processed/twitter/full/
by summing per-intervention Poisson-binomial distributions at q_CC=0.182.

Uses the same DuckDB-style per-interval aggregation that produces M/N_kappa_q0.192,
but at the lower q required for CC to match Monsted Fig 3.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_processing.utils import compute_exposure_probabilities_fast

Q_CC = 0.182
KAPPAMAX = 20
PROCESSED = Path("data/processed/twitter")
INTERVENTIONS = sorted(
    d.name for d in PROCESSED.iterdir() if d.is_dir() and d.name != "full"
)

print(f"Interventions: {INTERVENTIONS}")
print(f"Q_CC = {Q_CC}, KAPPAMAX = {KAPPAMAX}")

# Accumulate per-user Poisson-binomial distributions across interventions
# user_m[uid] = sum of P(kappa) vectors for ADOPTED users
# user_n[uid] = sum of P(kappa) vectors for NON-ADOPTED users
user_m: dict[int, np.ndarray] = {}
user_n: dict[int, np.ndarray] = {}

for iv in INTERVENTIONS:
    am = pd.read_parquet(PROCESSED / iv / "A_matrix.parquet")
    bv = pd.read_parquet(PROCESSED / iv / "B_vector.parquet")
    bot_cols = [c for c in am.columns if c != "user_id"]
    merged = am.merge(bv, on="user_id")
    vals = merged[bot_cols].fillna(0).values.astype(int)
    adopted = merged["adopted"].values.astype(bool)
    user_ids = merged["user_id"].values

    dist_cache: dict[str, np.ndarray] = {}
    for i in range(len(merged)):
        # A = sorted list of total-exposure-event counts per bot (already in A_matrix)
        A = sorted([int(v) for v in vals[i] if v > 0])
        key = str(A)
        if key not in dist_cache:
            p = compute_exposure_probabilities_fast(A, Q_CC)
            vec = np.zeros(KAPPAMAX, dtype=np.float64)
            for k, pv in p.items():
                if 1 <= k <= KAPPAMAX:
                    vec[k - 1] = pv
            dist_cache[key] = vec
        uid = int(user_ids[i])
        vec = dist_cache[key]
        if adopted[i]:
            if uid not in user_m:
                user_m[uid] = np.zeros(KAPPAMAX, dtype=np.float64)
            user_m[uid] += vec
        else:
            if uid not in user_n:
                user_n[uid] = np.zeros(KAPPAMAX, dtype=np.float64)
            user_n[uid] += vec
    print(f"  {iv}: {len(merged)} users, cache size={len(dist_cache)}")

# Build DataFrames
kappa_cols = [f"kappa_{k}" for k in range(1, KAPPAMAX + 1)]

m_rows = [[uid] + list(vec) for uid, vec in sorted(user_m.items())]
n_rows = [[uid] + list(vec) for uid, vec in sorted(user_n.items())]

m_df = pd.DataFrame(m_rows, columns=["user_id"] + kappa_cols)
n_df = pd.DataFrame(n_rows, columns=["user_id"] + kappa_cols)

out = PROCESSED / "full"
out.mkdir(parents=True, exist_ok=True)
m_df.to_parquet(out / f"M_kappa_q{Q_CC}.parquet", index=False)
n_df.to_parquet(out / f"N_kappa_q{Q_CC}.parquet", index=False)
print(f"\nSaved M_kappa_q{Q_CC}.parquet: {len(m_df)} adopted users")
print(f"Saved N_kappa_q{Q_CC}.parquet: {len(n_df)} non-adopted users")
