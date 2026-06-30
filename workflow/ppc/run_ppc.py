import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.simulation import (
    build_graph,
    build_kernel_bank,
    assign_kernels,
    build_initial_state,
    heterogeneous_contagion_process,
    get_activations,
    simple_kernel,
)
from src.inference import fit_stan, get_posterior_draws
from src.analysis import ppc_complexity

RES_DIR = ROOT / "results" / "ppc"

# DGP for OBSERVED data
RHO, GAMMA, TMAX = 0.125, 0.1, 1000
GRAPH = {"type": "edgelist", "params": {"filename": ROOT / "congress.edgelist"}}
KERNELS = [{"type": "simple", "beta": 0.01},
           {"type": "simple", "beta": 0.05}]
ASSIGNMENT = {"mode": "block_flip", "omega": 0.05,
              "communities": np.loadtxt(ROOT / "congress_communities.txt")}

# sampling/replication settings
SAMPLING = {"iter_sampling": 500, "iter_warmup": 1500}
NREP = 100
REG  = 1e-6

# seeds (fixed so a crash/resume reconstructs the SAME setup)
LABEL_SEED = 11       # kernel-to-node assignment for the observed mixture
INIT_SEED  = 17       # initial infected set, shared by observed + every null
OBS_SEED   = 13       # observed contagion realization

NP_CONFIG = ROOT / "configs" / "nonparametric_eb.yaml"
SM_CONFIG = ROOT / "configs" / "simple.yaml"


def _save_npz(fp: Path, **arrays):
    """Atomic save: write to a temp file object (no .npz auto-append), then rename."""
    fp.parent.mkdir(parents=True, exist_ok=True)
    tmp = fp.parent / (fp.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(tmp, fp)


def fit_np_mu(M, N):
    res = fit_stan(M, N, NP_CONFIG, instance_params={"K": 1, "sampling": {**SAMPLING}})
    return get_posterior_draws(res, "mu")[:, 0, :]


def main():
    RES_DIR.mkdir(parents=True, exist_ok=True)

    # fixed setup (deterministic given the seeds)
    adj, kmax = build_graph(GRAPH)
    n = adj.shape[0]
    x0 = build_initial_state(n, RHO, seed=INIT_SEED) # fixed initial infected set

    # observed: simulate the true (mixture) kernel, fit np + simple
    obs_ckpt = RES_DIR / "observed.npz"
    if obs_ckpt.exists():
        d = np.load(obs_ckpt)
        mus, beta_hat = d["mus"], float(d["beta_hat"])
        print(f"[obs] loaded checkpoint (beta_hat={beta_hat:.6f})")
    else:
        bank   = build_kernel_bank(KERNELS, kmax)
        labels = assign_kernels(n, ASSIGNMENT, len(KERNELS), seed=LABEL_SEED)
        c      = np.stack([bank[labels[i]] for i in range(n)])
        x      = heterogeneous_contagion_process(adj, GAMMA, c, x0, tmax=TMAX, seed=OBS_SEED)
        M, N   = get_activations(np.transpose(x), adj, kmax)

        mus      = fit_np_mu(M, N)
        sm_res   = fit_stan(M, N, SM_CONFIG, instance_params={"K": 1, "sampling": {**SAMPLING}})
        beta_hat = float(np.asarray(get_posterior_draws(sm_res, "beta")).reshape(-1).mean())
        _save_npz(obs_ckpt, mus=mus, beta_hat=beta_hat)
        print(f"[obs] fit done (beta_hat={beta_hat:.6f})")

    # null kernel is homogeneous simple at beta_hat -> compute curve once
    c_null = np.broadcast_to(simple_kernel(beta_hat, kmax), (n, kmax))

    # nulls: same graph + same x0, vary only the contagion seed
    null_dir = RES_DIR / "nulls"
    null_dir.mkdir(exist_ok=True)
    for r in range(NREP):
        fp = null_dir / f"null_{r:04d}.npz"
        if fp.exists():
            continue
        try:
            x_r    = heterogeneous_contagion_process(adj, GAMMA, c_null, x0, tmax=TMAX, seed=r)
            M_r, N_r = get_activations(np.transpose(x_r), adj, kmax)
            mu_r   = fit_np_mu(M_r, N_r)
            _save_npz(fp, mu_null=mu_r)
            print(f"[null {r:>3}/{NREP}] fit done")
        except Exception as e: # drop, don't crash
            print(f"[null {r:>3}/{NREP}] FAILED ({e}); skipping")

    # collect every completed null and score
    null_files = sorted(null_dir.glob("null_*.npz"))
    mu_rep_draws = [np.load(f)["mu_null"] for f in null_files]
    print(f"collected {len(mu_rep_draws)}/{NREP} null replicates")
    if not mu_rep_draws:
        raise RuntimeError("no null replicates completed; nothing to calibrate against")

    res = ppc_complexity(mus, mu_rep_draws, beta=beta_hat, reg=REG, plus_one=True)

    summary = {
        "p_value":     res["p_value"],
        "d_obs":       res["d_obs"],
        "n_null":      res["n_null"], # effective R (failed fits dropped)
        "n_ge":        res["n_ge"],
        "R_requested": NREP,
        "beta_hat":    beta_hat,
        "reg":         REG,
    }
    (RES_DIR / "ppc_result.json").write_text(json.dumps(summary, indent=2))
    _save_npz(RES_DIR / "ppc_null_distribution.npz",
              d_null=res["d_null"], d_obs=np.array(res["d_obs"]))
    print("\nPPC result:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()