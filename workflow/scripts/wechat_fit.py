"""Snakemake script: fit one K on every WeChat node (rule wechat_fit).

Saves draws_mu (S, K, kmax), draws_pi (S, K) and post_prob (V, K), the posterior mean responsibility.
logit_p is ordered but does not identify the components, so every draw is relabelled by ascending mu(1).
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.inference.inference import fit_stan, get_posterior_draws

fixed = dict(snakemake.params.fixed)
K = int(fixed["K"])
sampling = json.loads(fixed["sampling_json"])

d = np.load(snakemake.input[0])
result = fit_stan(
    d["M"], d["N"],
    ROOT / fixed["model_config"],
    {"K": K, "sampling": {**sampling, "seed": int(fixed["seed"])}},
)

mu = get_posterior_draws(result, "mu").reshape(-1, K, d["M"].shape[1])
pi = get_posterior_draws(result, "pi").reshape(-1, K)
post = get_posterior_draws(result, "post_prob").reshape(-1, d["M"].shape[0], K)

order = np.argsort(mu[:, :, 0], axis=1)
mu = np.take_along_axis(mu, order[:, :, None], axis=1)
pi = np.take_along_axis(pi, order, axis=1)
post = np.take_along_axis(post, order[:, None, :], axis=2)

np.savez_compressed(snakemake.output[0], draws_mu=mu, draws_pi=pi, post_prob=post.mean(axis=0))
