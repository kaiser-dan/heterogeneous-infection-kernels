from pathlib import Path

import pytest

from src.models import stan_registry as sr


def test_registry_has_canonical_tokens():
    expected = {
        "LOGIT_RHO_LOCATION", "LOGIT_RHO_SCALE",
        "LOGIT_RHO_L_LOCATION", "LOGIT_RHO_L_SCALE",
        "LOGIT_RHO_H_LOCATION", "LOGIT_RHO_H_SCALE",
        "LOGIT_P_LOCATION", "LOGIT_P_SCALE",
        "LOGIT_MU_LOCATION", "LOGIT_MU_SCALE",
        "PI_CONCENTRATION", "CONCENTRATION", "SIGMA_SCALE",
        "W_RATE", "K0_LOCATION", "K0_SCALE",
        "DEVIATION_SCALE", "REPLICATES",
    }
    assert expected.issubset(set(sr.TOKEN_TABLE.keys()))


def test_parse_data_block_returns_declared_names():
    stan_src = """
    data {
        int kmax;
        int V;
        array[V, kmax] real M;
        array[V, kmax] real N;
        real LOGIT_RHO_LOCATION;
        real<lower=0> LOGIT_RHO_SCALE;
        vector<lower=0>[K] DEVIATION_SCALE;
    }
    parameters { real x; }
    """
    names = sr.parse_data_block_names(stan_src)
    assert {"kmax", "V", "M", "N", "LOGIT_RHO_LOCATION", "LOGIT_RHO_SCALE", "DEVIATION_SCALE"} <= names


def test_resolve_plain():
    ctx = sr.ResolverContext(empirical_logits={}, fallback=0.0, kmax=10)
    assert sr.RESOLVERS["plain"](2.5, ctx) == 2.5


def test_resolve_kmax_fraction():
    ctx = sr.ResolverContext(empirical_logits={}, fallback=0.0, kmax=10)
    assert sr.RESOLVERS["kmax_fraction"](0.5, ctx) == 5.0


def test_resolve_logit_empirical_passes_through_floats():
    ctx = sr.ResolverContext(empirical_logits={"empirical_floor": -3.0}, fallback=-1.0, kmax=10)
    assert sr.RESOLVERS["logit_empirical"](-2.5, ctx) == -2.5


def test_resolve_logit_empirical_handles_named_string():
    ctx = sr.ResolverContext(empirical_logits={"empirical_floor": -3.0}, fallback=-1.0, kmax=10)
    assert sr.RESOLVERS["logit_empirical"]("empirical_floor", ctx) == -3.0


def test_build_stan_data_only_injects_declared_tokens():
    stan_src = """
    data {
        int kmax;
        array[V, kmax] real M;
        array[V, kmax] real N;
        real LOGIT_RHO_LOCATION;
        real<lower=0> LOGIT_RHO_SCALE;
    }
    """
    ctx = sr.ResolverContext(empirical_logits={}, fallback=-10.0, kmax=20)
    config_model = {"logit_rho_location": -8.0, "logit_rho_scale": 1.5, "concentration": 99.0}
    extra = sr.build_stan_data_extras(stan_src, config_model, ctx)
    assert extra == {"LOGIT_RHO_LOCATION": -8.0, "LOGIT_RHO_SCALE": 1.5}


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_every_stan_data_block_token_has_table_entry():
    stan_files = list((REPO_ROOT / "src" / "models").glob("*.stan"))
    assert stan_files, "no Stan files found"
    unknown = set()
    for f in stan_files:
        names = sr.parse_data_block_names(f.read_text())
        for n in names:
            if n.isupper() and n not in sr.TOKEN_TABLE and n != "REPLICATES" and n != "DEVIATION_SCALE":
                unknown.add((f.name, n))
    assert not unknown, f"Stan tokens missing from registry: {unknown}"
