"""Snakemake script: build the observed PPC fit (rule ppc_observed, wildcard omega)."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.simulation import (
    build_graph, build_kernel_bank, assign_kernels,
    build_initial_state, heterogeneous_contagion_process, get_activations,
)
from src.inference import fit_stan, get_posterior_draws

fixed = dict(snakemake.params.fixed)
kernels = json.loads(fixed["kernels_json"])
sampling = json.loads(fixed["sampling_json"])

graph_cfg = {
    "type": fixed["graph_type"],
    "params": {"filename": ROOT / fixed["graph_filename"]},
}
adj, kmax = build_graph(graph_cfg)
n = adj.shape[0]
x0 = build_initial_state(n, fixed["rho"], seed=int(fixed["seed_init"]))

communities = np.loadtxt(ROOT / fixed["assignment_communities_file"])
assignment_cfg = {"mode": fixed["assignment_mode"], "omega": fixed["omega"],
                   "communities": communities}

bank = build_kernel_bank(kernels, kmax)
labels = assign_kernels(n, assignment_cfg, len(kernels), seed=int(fixed["seed_label"]))
c = np.stack([bank[labels[i]] for i in range(n)])
x = heterogeneous_contagion_process(adj, fixed["gamma"], c, x0, tmax=int(fixed["tmax"]), random_seed=int(fixed["seed_obs"]))
M, N = get_activations(np.transpose(x), adj, kmax)

np_res = fit_stan(M, N, ROOT / fixed["np_config"], {"K": 1, "sampling": sampling})
mus = get_posterior_draws(np_res, "mu")[:, 0, :]

sm_res = fit_stan(M, N, ROOT / fixed["simple_config"], {"K": 1, "sampling": sampling})
betas = np.asarray(get_posterior_draws(sm_res, "beta")).reshape(-1)
beta_hat = float(betas.mean())

np.savez_compressed(snakemake.output[0], mus=mus, beta_hat=np.array(beta_hat), betas=betas)
print(f"[obs omega={fixed['omega']}] fit done (beta_hat={beta_hat:.6f})")
