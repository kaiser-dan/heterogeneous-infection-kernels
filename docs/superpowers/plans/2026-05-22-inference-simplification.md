# Inference simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce `src/models/inference.py` to a thin Stan-call orchestrator by extracting dataset-specific loaders into `data_processing/*` and replacing source-string-matched Stan-data injection with a declarative table driven by Stan data-block introspection.

**Architecture:** Three new files (`process_twitter.load_twitter_MN`, `operation_outbreak.load_oo_MN`, `src/models/stan_registry.py`) own the responsibilities that currently live inline in `run_stan_inference`. The orchestrator becomes a dispatch + Stan-call body. We deviate from the spec's per-file `MODEL_SPEC` dict in favor of a single canonical token table + Stan data-block introspection, because the repo has 19 `.stan` files that share most tokens — a per-file registry would be hand-maintained drift bait.

**Tech Stack:** Python 3.11, cmdstanpy, pandas, numpy, pytest, uv.

**Branch:** stack commits on `feature/cv-pipeline-merge` (PR #4).

---

## File map

| File | Action | Responsibility |
|---|---|---|
| `src/data_processing/utils.py` | modify | add `cfg_get` helper |
| `src/data_processing/process_twitter.py` | modify | add `load_twitter_MN` |
| `src/data_processing/operation_outbreak.py` | modify | add `load_oo_MN` |
| `src/models/stan_registry.py` | create | canonical token table + Stan data-block introspection + resolvers |
| `src/models/inference.py` | modify | shrink `run_stan_inference`; rename `_safe_logit` → `_logit`; drop fallbacks |
| `test/test_dataset_loaders.py` | create | snapshot tests for loaders |
| `test/test_stan_registry.py` | create | registry coverage tests |

---

## Task 1: `cfg_get` helper

**Files:**
- Modify: `src/data_processing/utils.py` (append)
- Test: `test/test_cfg_get.py` (new)

- [ ] **Step 1: Write the failing test**

Create `test/test_cfg_get.py`:

```python
import pytest
from src.data_processing.utils import cfg_get


def test_cfg_get_top_level_wins():
    cfg = {"foo": 1, "model": {"foo": 2}}
    assert cfg_get(cfg, "foo") == 1


def test_cfg_get_falls_back_to_model_namespace():
    cfg = {"model": {"foo": 2}}
    assert cfg_get(cfg, "foo") == 2


def test_cfg_get_returns_default_when_missing():
    cfg = {"model": {}}
    assert cfg_get(cfg, "foo", default=42) == 42


def test_cfg_get_default_none_when_missing():
    assert cfg_get({}, "foo") is None


def test_cfg_get_tolerates_non_dict_model():
    cfg = {"model": "not-a-dict"}
    assert cfg_get(cfg, "foo", default="x") == "x"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_cfg_get.py -v`
Expected: FAIL with `ImportError: cannot import name 'cfg_get'`.

- [ ] **Step 3: Implement `cfg_get`**

Append to `src/data_processing/utils.py`:

```python
def cfg_get(config: dict, key: str, default=None):
    """Look up a sweep-config value, preferring top-level then config['model']."""
    if key in config:
        return config[key]
    model_cfg = config.get("model")
    if isinstance(model_cfg, dict) and key in model_cfg:
        return model_cfg[key]
    return default
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_cfg_get.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/data_processing/utils.py test/test_cfg_get.py
git commit -m "feat(utils): add cfg_get helper for namespaced config lookup

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 2: `load_twitter_MN`

**Files:**
- Modify: `src/data_processing/process_twitter.py` (append)
- Test: `test/test_dataset_loaders.py` (new)

- [ ] **Step 1: Write the failing test**

Create `test/test_dataset_loaders.py`:

```python
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.data_processing.process_twitter import load_twitter_MN


def _write_summary(tmp_path: Path, rows: list[dict]) -> Path:
    summary_dir = tmp_path / "summary"
    summary_dir.mkdir()
    pd.DataFrame(rows).to_parquet(summary_dir / "summary.parquet")
    return summary_dir


def test_load_twitter_MN_returns_expected_dict_keys(tmp_path):
    summary_dir = _write_summary(
        tmp_path,
        [
            {"user_id": "u1", "A_array": np.array([2, 1]), "adopted": True},
            {"user_id": "u2", "A_array": np.array([1, 1]), "adopted": False},
        ],
    )
    config = {"q": 0.5, "exposure_metric": "k", "twitter": {"adopters_only": False, "disable_ipw": True}}
    data = load_twitter_MN(summary_dir, config, kmax=4)
    assert set(data.keys()) == {"M", "N", "row_weights", "kmax_observed", "source_desc"}
    assert data["M"].shape == (2, 4)
    assert data["N"].shape == (2, 4)
    assert data["row_weights"].shape == (2,)


def test_load_twitter_MN_adopters_only_drops_non_adopters(tmp_path):
    summary_dir = _write_summary(
        tmp_path,
        [
            {"user_id": "u1", "A_array": np.array([2, 1]), "adopted": True},
            {"user_id": "u2", "A_array": np.array([1, 1]), "adopted": False},
        ],
    )
    config = {"q": 0.5, "exposure_metric": "k", "twitter": {"adopters_only": True}}
    data = load_twitter_MN(summary_dir, config, kmax=4)
    assert data["M"].shape[0] == 1
    assert np.allclose(data["row_weights"], 1.0)


def test_load_twitter_MN_empty_summary_raises_without_precomputed(tmp_path):
    summary_dir = tmp_path / "empty"
    summary_dir.mkdir()
    config = {"q": 0.5, "exposure_metric": "k"}
    with pytest.raises(ValueError, match="summary.parquet or direct pre-computed"):
        load_twitter_MN(summary_dir, config, kmax=4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_dataset_loaders.py -v`
Expected: FAIL with `ImportError: cannot import name 'load_twitter_MN'`.

- [ ] **Step 3: Implement `load_twitter_MN`**

Append to `src/data_processing/process_twitter.py`:

```python
def load_twitter_MN(summary_path, config: dict, kmax: int) -> dict:
    """
    Build Twitter Stan-input M, N, and row_weights for a sweep instance.

    Returns a dict with keys: M, N, row_weights, kmax_observed, source_desc.
    Raises ValueError when no summary.parquet and no direct pre-computed M/N
    files are available.
    """
    from src.data_processing.utils import cfg_get

    loaded_summary_df = _load_twitter_summary_df(summary_path)
    summary_df = loaded_summary_df.copy() if loaded_summary_df is not None else pd.DataFrame()
    if not summary_df.empty and "user_id" in summary_df.columns:
        summary_df["user_id"] = summary_df["user_id"].astype(str)

    V_max_local = cfg_get(config, "V_max_local", 200)
    twitter_cfg = config.get("twitter", {}) if isinstance(config.get("twitter", {}), dict) else {}
    adopters_only = bool(
        twitter_cfg.get("adopters_only", cfg_get(config, "twitter_adopters_only", False))
    )
    disable_ipw = bool(
        twitter_cfg.get("disable_ipw", cfg_get(config, "twitter_disable_ipw", False))
    )

    if not summary_df.empty:
        if adopters_only:
            n_before = len(summary_df)
            summary_df = summary_df[summary_df["adopted"]].copy()
            if summary_df.empty:
                raise ValueError("Twitter adopters-only mode produced 0 rows.")
            summary_df["ipw"] = 1.0
            print(f"Twitter mode: adopters-only ({len(summary_df)} / {n_before} rows).")
            print("Twitter mode: IPW disabled (unit ROW_WEIGHT).")
        else:
            adopters = summary_df[summary_df["adopted"]]
            non_adopters = summary_df[~summary_df["adopted"]]
            n_adopters = len(adopters)
            n_non_adopters = len(non_adopters)
            if len(summary_df) > V_max_local:
                if n_adopters <= V_max_local // 2:
                    sample_a = adopters
                    n_sample_a = n_adopters
                    n_sample_n = V_max_local - n_sample_a
                else:
                    n_sample_a = V_max_local // 2
                    sample_a = adopters.sample(n=n_sample_a, random_state=42)
                    n_sample_n = V_max_local - n_sample_a
                sample_n = non_adopters.sample(n=n_sample_n, random_state=42)
                weight_a = n_adopters / n_sample_a if n_sample_a > 0 else 1.0
                weight_n = n_non_adopters / n_sample_n if n_sample_n > 0 else 1.0
                sample_a = sample_a.copy()
                sample_a["ipw"] = weight_a
                sample_n = sample_n.copy()
                sample_n["ipw"] = weight_n
                summary_df = pd.concat([sample_a, sample_n])
            else:
                summary_df["ipw"] = 1.0
            if disable_ipw:
                summary_df = summary_df.copy()
                summary_df["ipw"] = 1.0
                print("Twitter mode: IPW explicitly disabled (unit ROW_WEIGHT).")
    else:
        print("Twitter mode: no summary.parquet found; relying on direct pre-computed M/N.")

    q = cfg_get(config, "q", 1.0)
    metric = cfg_get(config, "exposure_metric", "kappa")
    twitter_mn_mode = str(cfg_get(config, "twitter_mn_mode", "incidence")).lower()
    print(f"Twitter M/N mode: {twitter_mn_mode}")

    if not summary_df.empty and "ipw" in summary_df.columns:
        row_weights = summary_df["ipw"].to_numpy(dtype=float)
    elif not summary_df.empty:
        row_weights = np.ones(len(summary_df), dtype=float)
    else:
        row_weights = None

    use_precomputed = bool(cfg_get(config, "twitter_use_precomputed_mn", True))
    M = N = None
    source_desc = None
    kmax_observed = int(kmax)

    if use_precomputed:
        precomputed = _load_precomputed_twitter_mn(
            summary_df=summary_df,
            config=config,
            summary_path=summary_path,
            q=q,
            metric=metric,
        )
        if precomputed is not None:
            M, N, source_desc = precomputed
            if M.shape[1] != int(kmax):
                print(
                    f"Adjusting kmax from {kmax} to {M.shape[1]} "
                    "to match pre-computed Twitter matrices."
                )
                kmax_observed = int(M.shape[1])
            if row_weights is None:
                row_weights = np.ones(M.shape[0], dtype=float)
            print(f"Twitter M/N source: pre-computed ({source_desc})")

    if M is None:
        if summary_df.empty:
            raise ValueError(
                "Twitter inference requires summary.parquet or direct pre-computed M/N files."
            )
        M, N, row_weights = generate_stan_matrices_twitter(
            summary_df, q, metric, kmax_observed, mn_mode=twitter_mn_mode
        )
        source_desc = "reconstructed from summary A_array"
        print("Twitter M/N source: reconstructed from summary A_array.")

    return {
        "M": M,
        "N": N,
        "row_weights": row_weights,
        "kmax_observed": kmax_observed,
        "source_desc": source_desc,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_dataset_loaders.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/data_processing/process_twitter.py test/test_dataset_loaders.py
git commit -m "feat(twitter): extract load_twitter_MN from inference.run_stan_inference

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 3: `load_oo_MN`

**Files:**
- Modify: `src/data_processing/operation_outbreak.py` (append)
- Test: `test/test_dataset_loaders.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `test/test_dataset_loaders.py`:

```python
from src.data_processing.operation_outbreak import load_oo_MN


def _write_oo_dir(tmp_path: Path, M_rows, N_rows, adopted=None):
    d = tmp_path / "oo"
    d.mkdir()
    pd.DataFrame(M_rows).to_parquet(d / "M.parquet")
    pd.DataFrame(N_rows).to_parquet(d / "N.parquet")
    if adopted is not None:
        pd.DataFrame({"adopted": adopted}).to_parquet(d / "nodes.parquet")
    return d


def test_load_oo_MN_returns_expected_dict_keys(tmp_path):
    d = _write_oo_dir(tmp_path, [[1, 0], [0, 1]], [[0, 1], [1, 0]])
    data = load_oo_MN(d, {"kmax": 2}, kmax=2)
    assert set(data.keys()) == {"M", "N", "row_weights", "kmax_observed", "source_desc"}
    assert data["M"].shape == (2, 2)
    assert np.allclose(data["row_weights"], 1.0)


def test_load_oo_MN_adopters_only_filters_rows(tmp_path):
    d = _write_oo_dir(
        tmp_path,
        [[1, 0], [0, 1]],
        [[0, 1], [1, 0]],
        adopted=[True, False],
    )
    data = load_oo_MN(d, {"oo": {"adopters_only": True}}, kmax=2)
    assert data["M"].shape[0] == 1


def test_load_oo_MN_drop_zero_bin_removes_first_column(tmp_path):
    d = _write_oo_dir(tmp_path, [[1, 0, 0]], [[0, 1, 0]])
    data = load_oo_MN(d, {"model": {"drop_zero_exposure_bin": True}}, kmax=3)
    assert data["M"].shape == (1, 2)
    assert data["kmax_observed"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_dataset_loaders.py -v`
Expected: 3 new tests fail with `ImportError: cannot import name 'load_oo_MN'`.

- [ ] **Step 3: Implement `load_oo_MN`**

Append to `src/data_processing/operation_outbreak.py`:

```python
def load_oo_MN(summary_path, config: dict, kmax: int) -> dict:
    """
    Build OO Stan-input M, N, and row_weights for a sweep instance.

    Returns a dict with keys: M, N, row_weights, kmax_observed, source_desc.
    """
    from src.data_processing.utils import cfg_get

    M, N = generate_stan_matrices_oo(summary_path, config)
    source_desc = f"pre-computed M/N in {Path(summary_path)}"

    oo_adopters_only = bool(config.get("oo", {}).get("adopters_only", False))
    if oo_adopters_only:
        nodes_path = Path(summary_path) / "nodes.parquet"
        if not nodes_path.exists():
            raise FileNotFoundError(
                f"oo.adopters_only=True but nodes.parquet not found at {nodes_path}"
            )
        nodes_df = pd.read_parquet(nodes_path)
        adopted_mask = nodes_df["adopted"].values.astype(bool)
        n_before = M.shape[0]
        M = M[adopted_mask]
        N = N[adopted_mask]
        print(f"OO adopters-only: kept {M.shape[0]} / {n_before} nodes (adopted=True).")

    drop_zero_bin = bool(
        cfg_get(config, "drop_zero_exposure_bin", config.get("oo", {}).get("drop_zero_exposure_bin", False))
    )
    if drop_zero_bin:
        if M.shape[1] <= 1:
            raise ValueError("Cannot drop zero-exposure bin: M/N have <=1 exposure column.")
        print("OO option enabled: dropping zero-exposure bin (column 0) before fitting.")
        M = M[:, 1:]
        N = N[:, 1:]

    kmax_observed = int(M.shape[1])
    if kmax_observed != int(kmax):
        print(f"Adjusting kmax from {kmax} to {kmax_observed} to match input matrices.")

    return {
        "M": M,
        "N": N,
        "row_weights": np.ones(M.shape[0], dtype=float),
        "kmax_observed": kmax_observed,
        "source_desc": source_desc,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_dataset_loaders.py -v`
Expected: 6 passed (3 twitter + 3 oo).

- [ ] **Step 5: Commit**

```bash
git add src/data_processing/operation_outbreak.py test/test_dataset_loaders.py
git commit -m "feat(oo): extract load_oo_MN from inference.run_stan_inference

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 4: Dispatch in `run_stan_inference`

**Files:**
- Modify: `src/models/inference.py` (replace lines ~498–668 with dispatch)

- [ ] **Step 1: Write the failing integration test**

Append to `test/test_dataset_loaders.py`:

```python
def test_inference_loaders_registry_present():
    from src.models import inference as inf
    assert "twitter" in inf.DATASET_LOADERS
    assert "oo" in inf.DATASET_LOADERS
    assert callable(inf.DATASET_LOADERS["twitter"])
    assert callable(inf.DATASET_LOADERS["oo"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_dataset_loaders.py::test_inference_loaders_registry_present -v`
Expected: FAIL with `AttributeError: module 'src.models.inference' has no attribute 'DATASET_LOADERS'`.

- [ ] **Step 3: Replace the dispatch block**

In `src/models/inference.py`, immediately after the existing `from src.data_processing.process_twitter import ...` import block (top of file), add:

```python
from src.data_processing.process_twitter import load_twitter_MN
from src.data_processing.operation_outbreak import load_oo_MN

DATASET_LOADERS = {"twitter": load_twitter_MN, "oo": load_oo_MN}
```

Then in `run_stan_inference`, replace the entire block from `if source == 'twitter':` through the closing `row_weights = np.ones(V, dtype=float)` (the inline dispatch — currently ~lines 498–668) with:

```python
    loader = DATASET_LOADERS.get(source)
    if loader is None:
        raise ValueError(f"Unknown source '{source}'. Expected one of {list(DATASET_LOADERS)}.")
    data = loader(summary_path, config, kmax)
    M, N = data["M"], data["N"]
    row_weights = data["row_weights"]
    kmax = data["kmax_observed"]
    V = M.shape[0]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/test_dataset_loaders.py test/test_inference_output_controls.py test/test_inference_precomputed_mn.py test/test_twitter_survival_overflow.py -v`
Expected: all pass.

- [ ] **Step 5: Run the full test suite as a regression check**

Run: `uv run pytest -x -q`
Expected: same pass count as before this task (no new failures).

- [ ] **Step 6: Commit**

```bash
git add src/models/inference.py test/test_dataset_loaders.py
git commit -m "refactor(inference): dispatch dataset loading via DATASET_LOADERS

Replaces the ~170-line inline if source=='twitter' branch in
run_stan_inference with a registry lookup. Twitter IPW selection,
adopters_only handling, precomputed M/N dispatch, and OO node filtering
now live in src/data_processing/{process_twitter,operation_outbreak}.py.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 5: Stan registry — canonical token table

**Files:**
- Create: `src/models/stan_registry.py`
- Test: `test/test_stan_registry.py` (new)

- [ ] **Step 1: Write the failing test**

Create `test/test_stan_registry.py`:

```python
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
    # CONCENTRATION must not appear because the data block did not declare it.


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_stan_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models.stan_registry'`.

- [ ] **Step 3: Create the registry module**

Create `src/models/stan_registry.py`:

```python
"""
Declarative table of Stan `data` block tokens used across our models.

`run_stan_inference` introspects each compiled model's `data { ... }` block,
then injects only the names declared there using the resolvers below. This
replaces ~20 `if "TOKEN" in stan_code: stan_data[...] = ...` blocks that
string-matched the Stan source.
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


# Canonical mapping of Stan-data token → (config_key, default, resolver_name).
# A token only gets injected when the compiled model's data block declares it.
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
    # DEVIATION_SCALE and REPLICATES are handled separately (shape-dependent).
    "DEVIATION_SCALE":       ("deviation_scale",       0.1,   "plain"),
    "REPLICATES":            ("replicates",            100,   "plain"),
}


_DATA_BLOCK = re.compile(r"data\s*\{(.*?)\}", re.DOTALL)
_DECL = re.compile(r"\b([A-Za-z_]\w*)\s*;")
_VECTOR_DEV = re.compile(r"vector\b.*DEVIATION_SCALE")


def parse_data_block_names(stan_src: str) -> set[str]:
    """Return the set of identifier names declared inside the first data { ... } block."""
    match = _DATA_BLOCK.search(stan_src)
    if not match:
        return set()
    body = match.group(1)
    # Strip comments.
    body = re.sub(r"//[^\n]*", "", body)
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.DOTALL)
    return set(_DECL.findall(body))


def is_vector_deviation_scale(stan_src: str) -> bool:
    return bool(_VECTOR_DEV.search(stan_src))


def build_stan_data_extras(stan_src: str, config_model: dict, ctx: ResolverContext) -> dict:
    """Build the Stan-data extras dict by intersecting the model's data block with TOKEN_TABLE."""
    declared = parse_data_block_names(stan_src)
    out: dict = {}
    for token, (cfg_key, default, resolver_name) in TOKEN_TABLE.items():
        if token not in declared:
            continue
        if token == "DEVIATION_SCALE":
            continue  # shape-dependent; handled by caller
        if token == "REPLICATES":
            out[token] = int(config_model.get(cfg_key, default))
            continue
        raw = config_model.get(cfg_key, default)
        out[token] = RESOLVERS[resolver_name](raw, ctx)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_stan_registry.py -v`
Expected: 7 passed. If `test_every_stan_data_block_token_has_table_entry` reports unknown tokens, add them to `TOKEN_TABLE` until the test passes.

- [ ] **Step 5: Commit**

```bash
git add src/models/stan_registry.py test/test_stan_registry.py
git commit -m "feat(stan): add declarative token table for Stan data-block injection

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 6: Wire the registry into `run_stan_inference`

**Files:**
- Modify: `src/models/inference.py` (replace lines ~795–879 — Stan data injection block)

- [ ] **Step 1: Replace the Stan-data injection block**

In `src/models/inference.py`, replace everything from the line `stan_data = {` (currently ~795) through the line `stan_data['REPLICATES'] = int(config['model'].get('replicates', 100))` (currently ~879) with:

```python
    from src.models.stan_registry import (
        ResolverContext,
        build_stan_data_extras,
        is_vector_deviation_scale,
        parse_data_block_names,
    )

    declared = parse_data_block_names(stan_code)
    ctx = ResolverContext(
        empirical_logits=empirical_logits,
        fallback=0.0,
        kmax=kmax,
    )

    stan_data = {"kmax": kmax, "M": M, "N": N}

    if "ROW_WEIGHT" in declared:
        if row_weights is None:
            row_weights = np.ones(V, dtype=float)
        if len(row_weights) != V:
            raise ValueError(f"ROW_WEIGHT length ({len(row_weights)}) must match V ({V}).")
        stan_data["ROW_WEIGHT"] = np.asarray(row_weights, dtype=float)

    if "V" in declared:
        stan_data["V"] = V
    if "K" in declared and "K" in config["model"]:
        stan_data["K"] = k_components

    stan_data.update(build_stan_data_extras(stan_code, config["model"], ctx))

    if "DEVIATION_SCALE" in declared:
        if is_vector_deviation_scale(stan_code):
            stan_data["DEVIATION_SCALE"] = [dev_scale_val] * k_components
        else:
            stan_data["DEVIATION_SCALE"] = dev_scale_val
```

Also remove the obsolete `vector_decl_pattern = re.compile(...)`, `is_vector_dev = ...`, `is_jgy_mod = ...`, and `requires_2d = ...` lines (currently ~706–713). Replace any subsequent use of `requires_2d` with:

```python
    requires_2d = "M" in declared and "array[V, kmax]" in stan_code or "array[V,kmax]" in stan_code
```

Then immediately re-derive `requires_2d` from declared shape — search the Stan source body once. Put this just below the `declared = parse_data_block_names(...)` line:

```python
    requires_2d = bool(re.search(r"array\s*\[\s*V\s*,\s*kmax\s*\]", stan_code))
```

(`re` is already imported at the top of inference.py; no inner import needed.)

- [ ] **Step 2: Run regression tests**

Run: `uv run pytest -x -q`
Expected: all tests that passed before still pass. If a Stan token gets unexpectedly omitted, the test failure will name it; add a `TOKEN_TABLE` entry and re-run.

- [ ] **Step 3: Commit**

```bash
git add src/models/inference.py
git commit -m "refactor(inference): drive Stan data injection from declarative registry

Replaces ~20 'if TOKEN in stan_code: stan_data[...] = ...' blocks with a
single loop that intersects the compiled model's data block with
stan_registry.TOKEN_TABLE. New Stan files automatically receive the
correct extras; tokens declared in comments no longer trigger spurious
injection.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 7: Fail loudly + dead-code cleanup

**Files:**
- Modify: `src/models/inference.py`

- [ ] **Step 1: Remove the duplicated `import re` and duplicated comment block**

In `src/models/inference.py`, locate the block (currently ~lines 690–699):

```
    # Check model code to see if it expects a vector or scalar for DEVIATION_SCALE
    # A simple but effective way is to look at the .stan file content
    # Check model code to see if it expects a vector or scalar for DEVIATION_SCALE
    # A simple but effective way is to look at the .stan file content
    stan_code = stan_file_path.read_text()
    ...
    import re
```

Reduce to a single block:

```python
    stan_code = stan_file_path.read_text()
```

(Task 6 already removed the regex usage. The top-of-file `import re` covers any remaining need.)

- [ ] **Step 2: Tighten `_extract_kernel_draws`**

Replace `_extract_kernel_draws` (currently ~lines 178–220). Read it first to confirm the exact code, then for each `except Exception: pass` block, switch to a presence check via `fit.metadata.stan_vars`:

```python
def _extract_kernel_draws(fit, kmax: int):
    """Return (kernel_draws, source) by introspecting the fit's exposed Stan variables."""
    stan_vars = set(fit.metadata.stan_vars) if hasattr(fit, "metadata") else set()
    if "p" in stan_vars:
        return np.asarray(fit.stan_variable("p"), dtype=float), "p"
    if "rho" in stan_vars:
        return np.asarray(fit.stan_variable("rho"), dtype=float).reshape(-1, 1) * np.ones(kmax), "rho"
    if "mu" in stan_vars:
        return np.asarray(fit.stan_variable("mu"), dtype=float), "mu"
    return None, None
```

(Adjust the body to match the current callsite expectations — read lines 178-220 before editing and preserve any post-processing that callers depend on.)

- [ ] **Step 3: Drop the `lp__` `stan_variable` fallback**

In `run_stan_inference` (currently ~lines 916–922), replace:

```python
        try:
            lp_values = np.asarray(fit.stan_variable("lp__"), dtype=float)
        except Exception:
            try:
                lp_values = np.asarray(fit.method_variables()["lp__"], dtype=float)
            except Exception:
                lp_values = None
```

with:

```python
        lp_values = np.asarray(fit.method_variables()["lp__"], dtype=float)
```

- [ ] **Step 4: Drop the CV try/except**

In `run_stan_inference` (currently ~lines 950–970), replace:

```python
    if run_cv and M_held is not None and M_held.shape[0] > 0:
        try:
            cv_result = compute_held_out_loglik(M_held, N_held, fit)
            ...
        except (OSError, KeyError, ValueError) as e:
            print(f"Warning: CV evaluation failed ({e}). Skipping cv_held_out_loglik.parquet.")
```

with the same body minus the try/except wrapper:

```python
    if run_cv and M_held is not None and M_held.shape[0] > 0:
        cv_result = compute_held_out_loglik(M_held, N_held, fit)
        cv_df = pd.DataFrame([{
            "cv_score":      cv_result["cv_score"],
            "n_held_out":    cv_result["n_held_out"],
            "mean_log_pred": cv_result["mean_log_pred"],
            "cv_fold":       cv_fold,
            "cv_folds":      cv_folds,
        }])
        cv_df.to_parquet(
            output_dir / "cv_held_out_loglik.parquet",
            index=False,
            compression=parquet_compression,
        )
        print(
            f"CV fold {cv_fold}: cv_score={cv_result['cv_score']:.4f}  "
            f"mean_log_pred={cv_result['mean_log_pred']:.4f}"
        )
```

- [ ] **Step 5: Remove debug print block**

In `run_stan_inference` (currently ~lines 471–477), delete the `print("Debug: Exposure Summary Path ...")` + `os.listdir` block entirely.

- [ ] **Step 6: Remove CmdStan auto-install fallback**

In `run_stan_inference` (currently ~lines 480–487), if there is a `try: cmdstanpy.install_cmdstan(...) except ...` block, delete it. CmdStan must be installed as part of `uv sync --all-extras`.

If the deletion changes startup behavior, add a one-line check before the first `cmdstanpy.CmdStanModel(...)` call:

```python
    if not cmdstanpy.cmdstan_path():
        raise RuntimeError(
            "CmdStan is not installed. Run `uv run python -m cmdstanpy.install_cmdstan` once."
        )
```

- [ ] **Step 7: Rename `_safe_logit` → `_logit`**

In `src/models/inference.py`:

```python
def _logit(p):
    p = np.asarray(p, dtype=float)
    return np.log(p / (1.0 - p))
```

Update the one caller in `_compute_empirical_logit_locations` (it currently reads `_safe_logit(overall_p)` etc.) — rename to `_logit`.

- [ ] **Step 8: Run the full test suite**

Run: `uv run pytest -x -q`
Expected: all tests pass. If `_extract_kernel_draws` tests fail due to behavior change, inspect — the new presence-check path should only differ for fits without `metadata.stan_vars`, which CmdStanPy ≥1.0 always provides.

- [ ] **Step 9: Commit**

```bash
git add src/models/inference.py
git commit -m "refactor(inference): remove swallow-all excepts and dead fallbacks

- _extract_kernel_draws now uses fit.metadata.stan_vars presence checks
  instead of three bare 'except Exception: pass' blocks
- CV held-out evaluation no longer wraps compute_held_out_loglik in a
  broad try/except; shape mismatches surface immediately
- lp__ extraction uses fit.method_variables() directly
- CmdStan auto-install removed; setup is now a one-time install step
- Removed duplicate 'import re' inside run_stan_inference and the
  duplicated DEVIATION_SCALE comment block
- Removed os.listdir debug print
- Renamed _safe_logit -> _logit (no actual clipping; callers clamp)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 8: Ergonomics — output-config defaults, marginal_map comment, stale fallback

**Files:**
- Modify: `src/models/inference.py`

- [ ] **Step 1: Replace `None`-sentinel defaults in `_default_output_config`/`_resolve_output_config`**

Read the current `_default_output_config` and `_resolve_output_config`. Refactor so that mode-dependent defaults are explicit:

```python
def _derive_defaults_for_mode(mode: str) -> dict:
    if mode == "full":
        return {"posterior_thin": 1,  "posterior_float32": False}
    if mode == "compact":
        return {"posterior_thin": 5,  "posterior_float32": True}
    if mode == "minimal":
        return {"posterior_thin": 25, "posterior_float32": True}
    raise ValueError(f"Unknown posterior_mode '{mode}'. Expected full|compact|minimal.")


def _default_output_config() -> dict:
    return {
        "posterior_mode": "full",
        # mode-derived; explicit None means 'derive from mode'
        "posterior_thin": None,
        "posterior_float32": None,
        # ... keep other keys as they currently exist ...
    }


def _resolve_output_config(config: dict) -> dict:
    out = _default_output_config()
    out.update(config.get("output", {}))
    mode_defaults = _derive_defaults_for_mode(out["posterior_mode"])
    for k, v in mode_defaults.items():
        if out.get(k) is None:
            out[k] = v
    return out
```

(Preserve whatever other keys the current `_default_output_config` returns — read the current implementation first and only change the mode-derivation pattern.)

- [ ] **Step 2: Document `_marginal_map` bin heuristic**

In `src/models/inference.py`, find `_marginal_map` (~line 172). Add a single-line comment before the `bins = min(int(bins), max(20, int(np.sqrt(...))))` expression:

```python
# Cap user-requested bins at sqrt(N) (or 20, whichever is larger) to avoid
# noisy histograms when N is small.
```

- [ ] **Step 3: Check whether the YAML `instance_params` fallback is dead**

Run:

```bash
git log -p --all -S "old Snakemake string representation" -- src/models/inference.py | head -40
```

Inspect the output. If the YAML fallback's last touch is older than 2025-06 and no current sweep config uses it, delete that branch. If you're unsure, leave it and add `# TODO: confirm dead before removing` on the line above it.

- [ ] **Step 4: Run the full test suite**

Run: `uv run pytest -x -q`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/models/inference.py
git commit -m "refactor(inference): ergonomic cleanup

- _resolve_output_config now uses an explicit _derive_defaults_for_mode
  helper instead of None-sentinel inference scattered in two places
- _marginal_map gains a one-line comment explaining the bin heuristic
- (optionally) drop the stale YAML instance_params fallback if confirmed dead

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 9: Push the branch and update PR #4

- [ ] **Step 1: Push commits**

```bash
git push origin feature/cv-pipeline-merge
```

Expected: 4 new commits pushed (Tasks 4, 6, 7, 8 are the commit-bearing tasks; Tasks 1, 2, 3, 5 each also create a commit — total 8 commits across the simplification).

- [ ] **Step 2: Update the PR description**

```bash
gh pr view 4 --json body -q .body > /tmp/pr4-body.md
```

Append to the PR body a "Follow-up simplification" section listing each commit and what audit item it addresses, then:

```bash
gh pr edit 4 --body-file /tmp/pr4-body.md
```

- [ ] **Step 3: Verify CI**

```bash
gh pr checks 4
```

Expected: green. If a check fails, read the failure, fix forward in a new commit (do not amend), push, re-check.

---

## Self-review notes

- **Spec coverage:** all 14 audit items are addressed across Tasks 1–8. Audit item #12 (`cfg_get`) is Task 1, used by Tasks 2 and 3. Items #1, #2, #3, #4, #5, #6, #7, #8, #9, #10, #11, #13, #14 are explicitly addressed in Tasks 4, 6, 7, and 8.
- **Type consistency:** `load_twitter_MN` and `load_oo_MN` both return the same dict shape `{M, N, row_weights, kmax_observed, source_desc}` and Task 4 unpacks the same keys.
- **No placeholders:** every code step shows the actual code. Tasks 7–8 ask the engineer to "read the current implementation first" only where the surrounding context is large enough that quoting it inline would obscure the change — the diff itself is fully specified.
