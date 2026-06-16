"""
Declarative table of Stan `data` block tokens for the two active models:
  - mixture_simple_contagion
  - non_parametric_monotonic_empirical_bayes

`build_stan_data_extras` introspects each compiled model's `data { ... }` block
and injects only the tokens declared there, resolved from config or defaults.

Tokens excluded from build_stan_data_extras (handled by caller in inference.py):
  - V, K, kmax: structural dimensions, set from data shape
  - M, N: array[V, kmax] count matrices, set from data
  - DEVIATION_SCALE: vector[K], assembled per-model in inference.py
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


RESOLVERS = {
    "plain": _resolve_plain,
    "logit_empirical": _resolve_logit_empirical,
}

# TOKEN_TABLE maps each Stan data-block token to:
#   (config_key, default_value, resolver_name)
#
# mixture_simple_contagion uses:
#   LOGIT_RHO_LOCATION, LOGIT_RHO_SCALE, PI_CONCENTRATION, SIGMA_SCALE
#
# non_parametric_monotonic_empirical_bayes uses:
#   CONCENTRATION, LOGIT_P_LOCATION, LOGIT_P_SCALE
#   (DEVIATION_SCALE is also declared but excluded — see module docstring)
TOKEN_TABLE: dict[str, tuple[str, float, str]] = {
    "LOGIT_RHO_LOCATION": ("logit_rho_location", -10.0, "logit_empirical"),
    "LOGIT_RHO_SCALE":    ("logit_rho_scale",      2.0, "plain"),
    "PI_CONCENTRATION":   ("pi_concentration",      1.0, "plain"),
    "SIGMA_SCALE":        ("sigma_scale",           1.0, "plain"),
    "CONCENTRATION":      ("concentration",         1.0, "plain"),
    "LOGIT_P_LOCATION":   ("logit_p_location",      0.0, "logit_empirical"),
    "LOGIT_P_SCALE":      ("logit_p_scale",         2.0, "plain"),
}

# Tokens declared in Stan data blocks but injected by the caller, not here.
_CALLER_TOKENS = frozenset({"V", "K", "M", "N", "kmax", "DEVIATION_SCALE"})

_DATA_BLOCK = re.compile(r"data\s*\{(.*?)\}", re.DOTALL)
_DECL = re.compile(r"\b([A-Za-z_]\w*)\s*;")


def parse_data_block_names(stan_src: str) -> set[str]:
    """Return identifiers declared inside the first data { ... } block."""
    match = _DATA_BLOCK.search(stan_src)
    if not match:
        return set()
    body = match.group(1)
    body = re.sub(r"//[^\n]*", "", body)
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.DOTALL)
    return set(_DECL.findall(body))


def build_stan_data_extras(stan_src: str, config_model: dict, ctx: ResolverContext) -> dict:
    """Build the Stan data extras dict for the given model source.

    Intersects the model's declared data-block tokens with TOKEN_TABLE,
    resolves each from config_model (falling back to defaults), and returns
    the result. Tokens in _CALLER_TOKENS are always skipped.
    """
    declared = parse_data_block_names(stan_src)
    out: dict = {}
    for token, (cfg_key, default, resolver_name) in TOKEN_TABLE.items():
        if token not in declared:
            continue
        if token in _CALLER_TOKENS:
            continue
        raw = config_model.get(cfg_key, default)
        out[token] = RESOLVERS[resolver_name](raw, ctx)
    return out
