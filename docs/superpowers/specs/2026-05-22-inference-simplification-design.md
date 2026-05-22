# Inference simplification design

**Date:** 2026-05-22
**Scope:** `src/models/inference.py` and adjacent data_processing modules
**Status:** approved, pending implementation plan

## Background

`src/models/inference.py` carried ~1000 lines of code with three classes of complexity layered on top of the actual Stan-call orchestration:

1. Dataset-specific data loading for Twitter and Operation Outbreak (OO).
2. ~20 near-identical `if "TOKEN" in stan_code: stan_data[key] = value` injection blocks driven by string-matching the Stan source.
3. Broad `except Exception: pass` swallowers, dead fallback paths, and small ergonomic decay.

A prior refactor moved leaf data-loading helpers (`_binomial_level_probs`, `_load_precomputed_twitter_mn`, `generate_stan_matrices_oo`, etc.) into `src/data_processing/*`. The orchestration in `run_stan_inference` still owns dataset semantics, so the abstraction leak persists.

A code-simplifier audit (2026-05-22) enumerated 14 items. This spec covers all 14, grouped into four commits, stacked on the open `feature/cv-pipeline-merge` branch (PR #4).

## Goals

1. `run_stan_inference` becomes a thin orchestrator with no `if source == "twitter"` branch.
2. Stan-data injection is driven by a per-model declarative registry, not source-string matching.
3. Swallow-all `except` blocks are removed; failures surface immediately.
4. Reduce `inference.py` from ~580 lines (post prior refactor) to ~300, all orchestration + Stan-call plumbing.

## Non-goals

- No changes to Stan model files themselves.
- No changes to the CV pipeline semantics (already covered by the just-merged work).
- No changes to PPC pipeline semantics.
- No introduction of new dependencies.

## Design

### Commit 1 — Dataset loaders extracted

New functions:

- `src/data_processing/process_twitter.py::load_twitter_MN(summary_path, config, kmax) -> dict`
  - Returns `{"M": ndarray, "N": ndarray, "row_weights": ndarray, "kmax_observed": int, "source_desc": str}`.
  - Owns: summary loading, IPW column selection, `adopters_only` conversion, `drop_zero_exposure_bin`, precomputed-vs-reconstructed M/N dispatch, `mn_mode` resolution.
- `src/data_processing/operation_outbreak.py::load_oo_MN(summary_path, config, kmax) -> dict`
  - Returns the same dict shape; `row_weights` is all-ones unless OO grows an IPW path in the future.

New helper in `src/data_processing/utils.py`:

- `cfg_get(config, key, default=None)` — checks top-level `config[key]` first, falls back to `config["model"][key]`. Replaces the repeated `config['model'].get(k, config.get(k, default))` pattern.

`run_stan_inference` dispatch shrinks to:

```python
LOADERS = {"twitter": load_twitter_MN, "oo": load_oo_MN}
data = LOADERS[source](summary_path, config, kmax)
M, N, row_weights = data["M"], data["N"], data["row_weights"]
```

Imports in `inference.py` collapse to the loader functions plus the existing `convert_incidence_to_adopters_only_survival` (now used internally by `load_twitter_MN`).

### Commit 2 — Stan-data declarative registry

New file: `src/models/stan_registry.py`.

```python
MODEL_SPEC = {
    "model_simple_contagion.stan": {
        "shape": "2D",                     # was: '"array[V, kmax]" in stan_code'
        "dev_scale": "vector",             # was: vector_decl_pattern match
        "replicates": False,               # was: '"REPLICATES" in stan_code'
        "data_keys": [
            ("LOGIT_RHO_LOCATION", "logit_rho_location", -10.0, "logit_empirical"),
            ("LOGIT_RHO_SCALE",    "logit_rho_scale",    1.5,   "plain"),
            # ...
        ],
    },
    # one entry per .stan file under src/models/
}

RESOLVERS = {
    "plain":            lambda raw, _ctx: float(raw),
    "logit_empirical":  lambda raw, ctx:  _resolve_logit_location(raw, ctx.empirical_logits, ctx.fallback),
    "kmax_fraction":    lambda raw, ctx:  float(raw) * ctx.kmax,
}
```

In `inference.py`:

```python
spec = MODEL_SPEC[stan_path.name]
for token, cfg_key, default, resolver_name in spec["data_keys"]:
    raw = cfg_get(config, cfg_key, default)
    stan_data[token] = RESOLVERS[resolver_name](raw, ctx)
```

`ctx` is a `ResolverContext` namedtuple `(empirical_logits, fallback, kmax)` built once per `run_stan_inference` call.

Registry lookup of an unknown Stan filename raises `KeyError` with a message pointing at this file — silent failure is impossible.

The `shape`, `dev_scale`, `replicates` flags replace the three remaining substring checks against `stan_code`.

### Commit 3 — Fail loudly + dead-code cleanup

Discrete changes:

| Audit # | Change | File:approx-line |
|---|---|---|
| #4 | Drop duplicate `import re` and duplicated comment block | inference.py ~699 |
| #5 | `_extract_kernel_draws`: replace `except Exception: pass` with explicit `fit.metadata`-based presence checks | inference.py ~178-220 |
| #6 | `_align_lp_to_kernel_draws`: use `fit.method_variables()["lp__"]` directly, drop `stan_variable("lp__")` fallback | inference.py ~916-922 |
| #8 | Remove `try/except` around `compute_held_out_loglik` | inference.py ~969 |
| #9 | Remove `os.listdir` debug print block | inference.py ~471-477 |
| #10 | Remove CmdStan auto-install fallback; add a clear error message pointing at `uv run python -m cmdstanpy.install_cmdstan` | inference.py ~480-487 |
| #11 | Rename `_safe_logit` → `_logit` (no clipping; callers already clamp) | inference.py ~37 |

### Commit 4 — Ergonomics

| Audit # | Change |
|---|---|
| #7 | Replace `None`-sentinel defaults in `_default_output_config` / `_resolve_output_config` with explicit `_derive_defaults_for_mode(mode)` helper |
| #13 | Remove stale YAML `instance_params` fallback after confirming via `git log` it's dead |
| #14 | Add one-line comment to `_marginal_map` explaining the `max(20, sqrt(N))` bin-count heuristic |

## Testing

Each commit runs `uv run pytest -x` before pushing.

New tests:

- **Commit 1:** `test/test_dataset_loaders.py::test_load_twitter_MN_matches_inline_legacy`. Build a fixture from one current sweep instance (run inference inline, snapshot M/N/row_weights), then assert `load_twitter_MN(...)` returns bit-identical arrays. One similar test for OO.
- **Commit 2:** `test/test_stan_registry.py::test_registry_covers_all_models`. Walks `src/models/*.stan`, asserts each has a `MODEL_SPEC` entry. Second test: `test_stan_registry_data_keys_match_compiled_stan_data`. Compile each Stan model and assert every declared `data { ... }` block name has a registry mapping.
- **Commits 3 and 4:** no new tests; rely on existing coverage in `test_inference_output_controls.py`, `test_inference_precomputed_mn.py`, `test_twitter_survival_overflow.py`, `test_synthetic_pipeline.py`.

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| CmdStan auto-install removal breaks first-time setup | Medium | Clear error pointing at install command; document in README |
| Stan registry misses a `.stan` file | Low (test enforces) | `test_registry_covers_all_models` |
| `load_twitter_MN` semantic drift from inline version | Medium | Snapshot test in commit 1 |
| PR #4 grows large | High | Commits clearly scoped; commit message names audit items |
| Removed broad-except surfaces latent bugs in `_extract_kernel_draws` | Medium | Acceptable — surfacing is the goal |

## Out of scope (deferred)

- Moving PPC orchestration out of `inference.py` (separate concern; not in audit).
- Migrating away from cmdstanpy.
- Touching the Stan files themselves.

## References

- Code-simplifier audit: subagent run 2026-05-22 (14 items).
- Prior refactor: commit moving Twitter/OO data-loading helpers into `src/data_processing/*` (this branch, pre-spec).
- PR: <https://github.com/kaiser-dan/heterogeneous-infection-kernels/pull/4> (feature/cv-pipeline-merge).
