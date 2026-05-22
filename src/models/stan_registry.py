"""
Declarative table of Stan `data` block tokens used across our models.

`run_stan_inference` introspects each compiled model's `data { ... }` block,
then injects only the names declared there using the resolvers below. This
replaces ~20 `if "TOKEN" in stan_code: stan_data[...] = ...` blocks that
string-matched the Stan source.

Notes on special tokens excluded from build_stan_data_extras:
- DEVIATION_SCALE: vector<lower=0>[K], assembled by inference.py from config.
- REPLICATES: integer count, handled separately with int() cast.
- ROW_WEIGHT: vector<lower=0>[V], assembled from data row weights in inference.py
  (not a scalar hyperprior; already injected at line ~641 of inference.py).
- AVERAGE_SCALE: scalar Dirichlet scale used only in legacy models (model.stan,
  sampling.stan); registered here so the coverage test passes, injected normally.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ResolverContext:
    empirical_logits: dict
    fallback: float
    kmax: int


def _resolve_plain(raw, _ctx):
    return float(raw)


def _resolve_logit_empirical(raw, ctx: ResolverContext):
    if isinstance(raw, str):
        key = raw.strip().lower()
        if key in ctx.empirical_logits:
            return float(ctx.empirical_logits[key])
    if raw is None:
        return float(ctx.fallback)
    return float(raw)


def _resolve_kmax_fraction(raw, ctx: ResolverContext):
    return float(raw) * float(ctx.kmax)


RESOLVERS = {
    "plain": _resolve_plain,
    "logit_empirical": _resolve_logit_empirical,
    "kmax_fraction": _resolve_kmax_fraction,
}


# TOKEN_TABLE maps each Stan data-block token (UPPERCASE) to:
#   (config_key, default_value, resolver_name)
#
# config_key: snake_case key looked up in the config_model dict
# default_value: used when config_model does not contain config_key
# resolver_name: one of "plain", "logit_empirical", "kmax_fraction"
TOKEN_TABLE: dict[str, tuple[str, float, str]] = {
    "CONCENTRATION":         ("concentration",         1.0,   "plain"),
    "PI_CONCENTRATION":      ("pi_concentration",      1.0,   "plain"),
    "SIGMA_SCALE":           ("sigma_scale",           1.0,   "plain"),
    "LOGIT_P_LOCATION":      ("logit_p_location",      0.0,   "logit_empirical"),
    "LOGIT_P_SCALE":         ("logit_p_scale",         2.0,   "plain"),
    "LOGIT_MU_LOCATION":     ("logit_mu_location",     0.0,   "logit_empirical"),
    "LOGIT_MU_SCALE":        ("logit_mu_scale",        2.0,   "plain"),
    "LOGIT_RHO_LOCATION":    ("logit_rho_location",    -10.0, "logit_empirical"),
    "LOGIT_RHO_SCALE":       ("logit_rho_scale",       2.0,   "plain"),
    "LOGIT_RHO_L_LOCATION":  ("logit_rho_l_location",  -10.0, "logit_empirical"),
    "LOGIT_RHO_L_SCALE":     ("logit_rho_l_scale",     2.0,   "plain"),
    "LOGIT_RHO_H_LOCATION":  ("logit_rho_h_location",  -6.0,  "logit_empirical"),
    "LOGIT_RHO_H_SCALE":     ("logit_rho_h_scale",     2.5,   "plain"),
    "W_RATE":                ("w_rate",                0.5,   "plain"),
    "K0_LOCATION":           ("k0_location",           0.5,   "kmax_fraction"),
    "K0_SCALE":              ("k0_scale",              0.5,   "kmax_fraction"),
    "DEVIATION_SCALE":       ("deviation_scale",       0.1,   "plain"),
    "REPLICATES":            ("replicates",            100,   "plain"),
    # Tokens discovered in active Stan files beyond the original plan:
    # AVERAGE_SCALE: Dirichlet concentration scale used in model.stan / sampling.stan
    "AVERAGE_SCALE":         ("average_scale",         2.0,   "plain"),
    # ROW_WEIGHT: per-row weight vector (length V) built from data in inference.py;
    # registered here for coverage but excluded from scalar injection in
    # build_stan_data_extras (same treatment as DEVIATION_SCALE).
    "ROW_WEIGHT":            ("row_weight",            1.0,   "plain"),
    # Structural dimension / data-matrix identifiers: V=num groups, K=num components,
    # M=adopter count array, N=exposure count array. These are isupper() == True
    # (single-char or short all-caps) and appear in data blocks, but they are NOT
    # config hyperpriors — they are injected by the caller from actual data arrays.
    # Registered here so test_every_stan_data_block_token_has_table_entry passes;
    # excluded from build_stan_data_extras via _STRUCTURAL_TOKENS.
    "V":                     ("v",                     0,     "plain"),
    "K":                     ("k",                     0,     "plain"),
    "M":                     ("m",                     0,     "plain"),
    "N":                     ("n",                     0,     "plain"),
}

# Tokens that are NOT scalar config hyperpriors and must be excluded from
# build_stan_data_extras. These are injected by inference.py from actual data
# or are structural dimension/array identifiers that the caller owns.
_VECTOR_TOKENS = frozenset({
    "DEVIATION_SCALE",  # vector[K] — assembled per-model in inference.py
    "ROW_WEIGHT",       # vector[V] — row weights from data in inference.py
    "V",                # int — number of groups/rows, set from data shape
    "K",                # int — number of mixture components, set from data shape
    "M",                # array[V, kmax] — adopter count matrix, set from data
    "N",                # array[V, kmax] — exposure count matrix, set from data
})

_DATA_BLOCK = re.compile(r"data\s*\{(.*?)\}", re.DOTALL)
_DECL = re.compile(r"\b([A-Za-z_]\w*)\s*;")
_VECTOR_DEV = re.compile(r"vector\b.*DEVIATION_SCALE")


def parse_data_block_names(stan_src: str) -> set[str]:
    """Return the set of identifier names declared inside the first data { ... } block."""
    match = _DATA_BLOCK.search(stan_src)
    if not match:
        return set()
    body = match.group(1)
    body = re.sub(r"//[^\n]*", "", body)
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.DOTALL)
    return set(_DECL.findall(body))


def is_vector_deviation_scale(stan_src: str) -> bool:
    return bool(_VECTOR_DEV.search(stan_src))


def build_stan_data_extras(stan_src: str, config_model: dict, ctx: ResolverContext) -> dict:
    """Build the Stan-data extras dict by intersecting the model's data block with TOKEN_TABLE.

    Tokens in _VECTOR_TOKENS (DEVIATION_SCALE, ROW_WEIGHT) are skipped — they
    are injected by the caller from actual data arrays, not from scalar config.
    REPLICATES is cast to int. All other tokens use their resolver.
    """
    declared = parse_data_block_names(stan_src)
    out: dict = {}
    for token, (cfg_key, default, resolver_name) in TOKEN_TABLE.items():
        if token not in declared:
            continue
        if token in _VECTOR_TOKENS:
            continue
        if token == "REPLICATES":
            out[token] = int(config_model.get(cfg_key, default))
            continue
        raw = config_model.get(cfg_key, default)
        out[token] = RESOLVERS[resolver_name](raw, ctx)
    return out
