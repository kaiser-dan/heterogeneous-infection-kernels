"""Snakemake script: fit one null replicate (rule ppc_null, wildcards omega, r)."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.simulation import build_graph, build_initial_state, heterogeneous_contagion_process, get_activations, simple_kernel
from src.inference import fit_stan, get_posterior_draws

fixed = dict(snakemake.params.fixed)
sampling = json.loads(fixed["sampling_json"])
r = int(snakemake.wildcards.r)

graph_cfg = {
    "type": fixed["graph_type"],
    "params": {"filename": ROOT / fixed["graph_filename"]},
}
adj, kmax = build_graph(graph_cfg)
n = adj.shape[0]
x0 = build_initial_state(n, fixed["rho"], seed=int(fixed["seed_null_init"]))

obs = np.load(snakemake.input[0]) # observed.npz, for beta_hat
beta_draws = obs["betas"]
rng = np.random.default_rng(int(fixed["seed_null_init"]))
beta_r = float(beta_draws[rng.integers(len(beta_draws))]) # one posterior draw per replicate
c_null = np.broadcast_to(simple_kernel(beta_r, kmax), (n, kmax))

x_r = heterogeneous_contagion_process(
    adj, float(fixed["gamma"]), c_null, x0,
    tmax=int(fixed["tmax"]), random_seed=int(fixed["seed_null"]),
)
M_r, N_r = get_activations(np.transpose(x_r), adj, kmax)

res = fit_stan(M_r, N_r, ROOT / fixed["np_config"], {"K": 1, "sampling": sampling})
mu_r = get_posterior_draws(res, "mu")[:, 0, :]

np.savez_compressed(snakemake.output[0], mu_null=mu_r, beta_null=beta_r)
print(f"[null omega={fixed['omega']} r={r}] fit done")
