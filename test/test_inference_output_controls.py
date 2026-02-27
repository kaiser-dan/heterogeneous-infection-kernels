"""
Test: aggregating M/N outputs from src/data_processing across interventions
reproduces apply_q_weighting from scripts/archive/monsted_fig3_reproduce.py.

The two code paths compute identical quantities via different routes:
  - reproduce script  : accumulates p_k in one pass  → sc_exposed[k], sc_shared[k]
  - process_twitter   : stores per-user p_dist in M/N columns, sums across nodes
Invariants tested:
  sum(M[k] across nodes) == sc_shared[k]     (expected adopters at level k)
  sum(M[k] + N[k] across nodes) == sc_exposed[k]  (total expected exposures at k)
"""

import importlib.util as _ilu
import numpy as np
from pathlib import Path

# ── Load distribution functions from reproduction script ─────────────────────
_SCRIPT = (
    Path(__file__).parent.parent
    / "scripts" / "archive" / "monsted_fig3_reproduce.py"
)
_spec = _ilu.spec_from_file_location("reproduce", _SCRIPT)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

apply_q_weighting             = _mod.apply_q_weighting
compute_binomial_distribution  = _mod.compute_binomial_distribution
compute_exposure_probabilities = _mod.compute_exposure_probabilities

# ── Synthetic exposure maps (2 interventions, 5 users total) ─────────────────
MAPS = [
    {
        "user2N_sc": {101: 2, 102: 3, 103: 1},
        "user2A_cc": {101: (2,), 102: (1, 2), 103: (1,)},
        "adopted":   {101},
    },
    {
        "user2N_sc": {201: 4, 202: 1},
        "user2A_cc": {201: (3, 1), 202: (1,)},
        "adopted":   {202},
    },
]

Q    = 0.20
KMAX = 7


def _mn_aggregate(maps, q, kmax):
    """
    Mimic process_twitter.save_matrices + aggregate_all_interventions:
      adopted user   → probability p_k goes into M column
      non-adopted    → probability p_k goes into N column
    Returns (sc_M, sc_N), (cc_M, cc_N) — each a dict {k: float}.
    """
    sc_M = {k: 0.0 for k in range(1, kmax + 1)}
    sc_N = {k: 0.0 for k in range(1, kmax + 1)}
    cc_M = {k: 0.0 for k in range(1, kmax + 1)}
    cc_N = {k: 0.0 for k in range(1, kmax + 1)}

    for iv in maps:
        adopted = iv["adopted"]

        for user, N in iv["user2N_sc"].items():
            p_dist = compute_binomial_distribution(N, q)
            target = sc_M if user in adopted else sc_N
            for k in range(1, kmax + 1):
                target[k] += p_dist.get(k, 0.0)

        for user, A_tuple in iv["user2A_cc"].items():
            p_dist = compute_exposure_probabilities(A_tuple, q)
            target = cc_M if user in adopted else cc_N
            for k in range(1, kmax + 1):
                target[k] += p_dist.get(k, 0.0)

    return (sc_M, sc_N), (cc_M, cc_N)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_aggregate_mn_sc_matches_reproduce_script():
    """SC: sum of M columns = sc_shared; sum of M+N columns = sc_exposed."""
    sc_ref, _ = apply_q_weighting(MAPS, Q)
    (sc_M, sc_N), _ = _mn_aggregate(MAPS, Q, KMAX)

    for k in range(1, KMAX + 1):
        ref_shared  = sc_ref["shared"].get(k, 0.0)
        ref_exposed = sc_ref["exposed"].get(k, 0.0)
        np.testing.assert_allclose(
            sc_M[k], ref_shared, rtol=1e-10,
            err_msg=f"SC shared (M) mismatch at k={k}",
        )
        np.testing.assert_allclose(
            sc_M[k] + sc_N[k], ref_exposed, rtol=1e-10,
            err_msg=f"SC exposed (M+N) mismatch at k={k}",
        )


def test_aggregate_mn_cc_matches_reproduce_script():
    """CC: sum of M columns = cc_shared; sum of M+N columns = cc_exposed."""
    _, cc_ref = apply_q_weighting(MAPS, Q)
    _, (cc_M, cc_N) = _mn_aggregate(MAPS, Q, KMAX)

    for k in range(1, KMAX + 1):
        ref_shared  = cc_ref["shared"].get(k, 0.0)
        ref_exposed = cc_ref["exposed"].get(k, 0.0)
        np.testing.assert_allclose(
            cc_M[k], ref_shared, rtol=1e-10,
            err_msg=f"CC shared (M) mismatch at k={k}",
        )
        np.testing.assert_allclose(
            cc_M[k] + cc_N[k], ref_exposed, rtol=1e-10,
            err_msg=f"CC exposed (M+N) mismatch at k={k}",
        )
