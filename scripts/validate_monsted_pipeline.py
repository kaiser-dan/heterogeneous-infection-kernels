"""
Validate pipeline outputs against Monsted et al. (2017) Figure 3 transcription.

SC uses q=0.192, CC uses q=0.18.
Panel A: P(adopted | k exposures)  — threshold |delta| <= 1.0 (1 pp)
Panel B: N_adopted at k exposures  — threshold |delta| <= 1.0
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

FULL = Path("data/processed/twitter/full")
TRANSCRIBED = Path("data/derived/monsted_fig3_transcribed.csv")

Q_SC = 0.192
Q_CC = 0.18   # pipeline stores 0.180 as 0.18 (Python float formatting)

target = pd.read_csv(TRANSCRIBED, comment="#")

def load_mn(metric, q):
    """Load M and N parquets, return (M_df, N_df) with columns user_id + k_1..k_max."""
    m = pd.read_parquet(FULL / f"M_{metric}_q{q}.parquet")
    n = pd.read_parquet(FULL / f"N_{metric}_q{q}.parquet")
    return m, n

def compute_panel(metric, q, col_prefix):
    """
    Returns dict: k -> (N_adopted, N_exposed, frac_adopted)
    N_adopted[k]  = sum over users of M[user, k]
    N_exposed[k]  = sum over users of (M[user, k] + N[user, k])
    frac_adopted  = N_adopted / N_exposed
    """
    M, N = load_mn(metric, q)
    # columns are like 'k_1', 'k_2', ... or 'kappa_1', 'kappa_2', ...
    exp_cols_m = [c for c in M.columns if c != "user_id"]
    exp_cols_n = [c for c in N.columns if c != "user_id"]
    assert exp_cols_m == exp_cols_n, "Column mismatch M vs N"

    M_sum = M[exp_cols_m].sum(axis=0)
    N_sum = N[exp_cols_n].sum(axis=0)
    total = M_sum + N_sum

    results = {}
    for col in exp_cols_m:
        # SC columns are bare '1','2',... ; CC columns are 'kappa_1','kappa_2',...
        parts = col.split("_")
        k = int(parts[-1])
        m_k = M_sum[col]
        t_k = total[col]
        frac = m_k / t_k if t_k > 0 else 0.0
        results[k] = (m_k, t_k, frac)
    return results

sc = compute_panel("k", Q_SC, "k")
cc = compute_panel("kappa", Q_CC, "kappa")

rows = []
for _, row in target.iterrows():
    panel = row["panel"]
    model = row["model"]
    k = int(row["x"])
    obs = float(row["observed"])

    if model == "SC":
        data = sc
    else:
        data = cc

    if k not in data:
        ours = float("nan")
        n_adopted = float("nan")
        n_exposed = float("nan")
    else:
        n_adopted, n_exposed, frac = data[k]
        if panel == "A":
            # Reproduce script stores pct_retweeted = frac * 100.
            # Figure Y-axis shows values like 0.010, 0.020 (i.e., fraction × 100),
            # so target=0.010 means 1.0% retweeted, and we must also be in those units.
            ours = frac * 100
        else:
            ours = n_adopted

    delta = ours - obs
    pct_err = 100 * delta / obs if obs != 0 else float("nan")

    # Dashboard.qmd treats Panel A deltas as percentage-point deltas after
    # multiplying by 100, so the pass/fail threshold is 1.0.
    # Panel B: absolute count, threshold |delta| <= 1.0.
    threshold = 1.0 if panel == "A" else 1.0
    passed = abs(delta) <= threshold
    rows.append({
        "panel": panel, "model": model, "k": k,
        "target": obs, "ours": round(ours, 5), "delta": round(delta, 5),
        "pct_err": round(pct_err, 1), "pass": "PASS" if passed else "FAIL"
    })

df = pd.DataFrame(rows)

print("=" * 80)
print(f"PIPELINE VS MONSTED FIG 3   (SC q={Q_SC}, CC q={Q_CC})")
print("=" * 80)
print(df.to_string(index=False))

fails = df[df["pass"] == "FAIL"]
print(f"\nTotal rows: {len(df)}  |  PASS: {len(df)-len(fails)}  |  FAIL: {len(fails)}")
if len(fails) > 0:
    worst = df.loc[df["delta"].abs().idxmax()]
    print(f"Worst: {worst['panel']},{worst['model']},k={worst['k']}: ours={worst['ours']} target={worst['target']} delta={worst['delta']}")
else:
    worst_row = df.loc[df["delta"].abs().idxmax()]
    print(f"Worst delta: {worst_row['panel']},{worst_row['model']},k={worst_row['k']}: delta={worst_row['delta']}")
    print("\nALL PASS")
