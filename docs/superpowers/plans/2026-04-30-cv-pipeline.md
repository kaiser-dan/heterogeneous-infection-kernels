# CV Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add k-fold cross-validation to the inference pipeline for selecting the number of mixture components K in the non-parametric contagion model.

**Architecture:** CV is opt-in via `cv_folds: k` in the config YAML and `cv_fold: [0..k-1]` as a sweep dimension; this produces k inference jobs per model config using the existing cartesian product logic. Each job holds out its fold's rows, fits on the rest, and computes held-out log-posterior-predictive-density (lppd) using the population-level kernels `mu[k]` and `pi` from the posterior. A separate aggregation step sums fold scores per model config.

**Tech Stack:** Python 3, CmdStanPy, NumPy, SciPy (`logsumexp`), Pandas, Snakemake 8.

---

## File Map

| File | Status | Responsibility |
|---|---|---|
| `src/models/inference.py` | Modify | Add `compute_held_out_loglik()`, fold-splitting logic, CV output write |
| `workflow/scripts/analysis/aggregate_cv.py` | Create | Sum fold CV scores, write `cv_summary.parquet` |
| `workflow/lib/experiment_plan.py` | Modify | Expose `cv_aggregate_outputs` from `WorkflowContext` |
| `workflow/rules/cv.smk` | Create | Snakemake `aggregate_cv` rule |
| `Snakefile` | Modify | Add `CV_AGGREGATE_OUTPUTS`, `cv_aggregate` target, include cv.smk |
| `test/test_cv_pipeline.py` | Create | Unit + integration tests for all new logic |

---

## Task 1: `compute_held_out_loglik` function

**Files:**
- Modify: `src/models/inference.py`
- Create: `test/test_cv_pipeline.py`

- [ ] **Step 1: Write the failing test**

Create `test/test_cv_pipeline.py`:

```python
import numpy as np
import pytest
from src.models.inference import compute_held_out_loglik


class MockFit:
    def __init__(self, mu_vals, pi_vals):
        self._vars = {"mu": np.asarray(mu_vals, dtype=float),
                      "pi": np.asarray(pi_vals, dtype=float)}

    def stan_variable(self, name):
        return self._vars[name]


def test_compute_held_out_loglik_k1_known_value():
    # K=1, kmax=3, S=2 draws (both identical), V_held=1
    # mu[s=0, k=0, :] = [0.1, 0.2, 0.3] for both draws
    # pi[s=0, k=0] = 1.0 for both draws
    # M_held = [[1, 0, 0]], N_held = [[0, 1, 0]]
    # Expected per-draw log-lik: log(pi=1) + 1*log(0.1) + 1*log(1-0.2)
    #                           = log(0.1) + log(0.8) = log(0.08)
    # log_pred = logsumexp([log(0.08), log(0.08)]) - log(2) = log(0.08)
    # cv_score = log(0.08)
    mu = np.array([[[0.1, 0.2, 0.3]], [[0.1, 0.2, 0.3]]])  # (S=2, K=1, kmax=3)
    pi = np.array([[1.0], [1.0]])                           # (S=2, K=1)
    fit = MockFit(mu, pi)

    M_held = np.array([[1.0, 0.0, 0.0]])  # (V=1, kmax=3)
    N_held = np.array([[0.0, 1.0, 0.0]])  # (V=1, kmax=3)

    result = compute_held_out_loglik(M_held, N_held, fit)

    expected_cv_score = np.log(0.08)
    assert abs(result["cv_score"] - expected_cv_score) < 1e-9
    assert result["n_held_out"] == 1
    assert abs(result["mean_log_pred"] - expected_cv_score) < 1e-9


def test_compute_held_out_loglik_k2_mixture():
    # K=2, kmax=2, S=1, V_held=1
    # Component 0: mu=[0.5, 0.5], pi=0.5
    # Component 1: mu=[0.5, 0.5], pi=0.5
    # Since both components identical, result = log(0.5 * p^M * (1-p)^N + 0.5 * p^M * (1-p)^N)
    # = log(p^M * (1-p)^N)
    mu = np.array([[[0.5, 0.5], [0.5, 0.5]]])   # (S=1, K=2, kmax=2)
    pi = np.array([[0.5, 0.5]])                   # (S=1, K=2)
    fit = MockFit(mu, pi)

    M_held = np.array([[1.0, 0.0]])
    N_held = np.array([[0.0, 1.0]])

    result = compute_held_out_loglik(M_held, N_held, fit)
    # log(0.5^1 * 0.5^1) = log(0.25)
    assert abs(result["cv_score"] - np.log(0.25)) < 1e-9


def test_compute_held_out_loglik_empty_held_out():
    mu = np.array([[[0.1, 0.2]]])  # (S=1, K=1, kmax=2)
    pi = np.array([[1.0]])
    fit = MockFit(mu, pi)

    M_held = np.zeros((0, 2))
    N_held = np.zeros((0, 2))

    result = compute_held_out_loglik(M_held, N_held, fit)
    assert result["cv_score"] == 0.0
    assert result["n_held_out"] == 0
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
uv run pytest test/test_cv_pipeline.py -v
```
Expected: `ImportError` or `AttributeError` — `compute_held_out_loglik` does not exist yet.

- [ ] **Step 3: Implement `compute_held_out_loglik` in `src/models/inference.py`**

Add this import near the top of `src/models/inference.py` (after existing imports):

```python
from scipy.special import logsumexp as _logsumexp
```

Add this function anywhere before `run_stan_inference`:

```python
def compute_held_out_loglik(
    M_held: np.ndarray,
    N_held: np.ndarray,
    fit,
) -> dict:
    """Log posterior predictive density for held-out nodes.

    Uses population-level kernels mu[k] and mixture weights pi from the
    posterior — no per-node latents required. Correct marginal predictive
    for a new node drawn from the population mixture.
    """
    eps = 1e-9

    if M_held.shape[0] == 0:
        return {"cv_score": 0.0, "n_held_out": 0, "mean_log_pred": np.nan}

    mu = np.asarray(fit.stan_variable("mu"), dtype=float)   # (S, K, kmax) or (S, kmax)
    pi = np.asarray(fit.stan_variable("pi"), dtype=float)   # (S, K) or (S,)

    # Normalise to (S, K, kmax) and (S, K) regardless of K=1 shape from CmdStanPy.
    if mu.ndim == 2:
        mu = mu[:, np.newaxis, :]   # (S, 1, kmax)
    if pi.ndim == 1:
        pi = pi[:, np.newaxis]      # (S, 1)

    S, K, kmax_model = mu.shape
    kmax_data = M_held.shape[1]
    k = min(kmax_model, kmax_data)

    mu_c = np.clip(mu[:, :, :k], eps, 1.0 - eps)    # (S, K, k)
    log_mu = np.log(mu_c)                             # (S, K, k)
    log_1m_mu = np.log(1.0 - mu_c)                   # (S, K, k)
    log_pi = np.log(np.clip(pi, eps, 1.0))            # (S, K)

    # Broadcast: (V, 1, 1, k) * (1, S, K, k) -> sum over k -> (V, S, K)
    M_b = M_held[:, np.newaxis, np.newaxis, :k]       # (V, 1, 1, k)
    N_b = N_held[:, np.newaxis, np.newaxis, :k]       # (V, 1, 1, k)
    log_mu_b = log_mu[np.newaxis, :, :, :]            # (1, S, K, k)
    log_1m_mu_b = log_1m_mu[np.newaxis, :, :, :]     # (1, S, K, k)

    ll_by_component = (M_b * log_mu_b + N_b * log_1m_mu_b).sum(axis=-1)  # (V, S, K)

    # Add log mixture weights and marginalise over K: (V, S)
    lps = ll_by_component + log_pi[np.newaxis, :, :]  # (V, S, K)
    log_lik = _logsumexp(lps, axis=-1)                 # (V, S)

    # Marginalise over posterior draws: (V,)
    log_pred = _logsumexp(log_lik, axis=-1) - np.log(S)  # (V,)

    cv_score = float(log_pred.sum())
    V_held = M_held.shape[0]
    return {
        "cv_score": cv_score,
        "n_held_out": V_held,
        "mean_log_pred": cv_score / V_held,
    }
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest test/test_cv_pipeline.py -v
```
Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/models/inference.py test/test_cv_pipeline.py
git commit -m "feat: add compute_held_out_loglik for CV lppd evaluation"
```

---

## Task 2: Fold splitting and CV output in `run_stan_inference`

**Files:**
- Modify: `src/models/inference.py`
- Modify: `test/test_cv_pipeline.py`

- [ ] **Step 1: Write the failing integration test**

Append to `test/test_cv_pipeline.py`:

```python
def test_fold_split_mask():
    """Modulo assignment: fold i gets rows where idx % cv_folds == i."""
    from src.models.inference import _cv_fold_mask
    V = 10
    cv_folds = 3
    masks = [_cv_fold_mask(V, cv_folds, fold) for fold in range(cv_folds)]

    # Each node assigned to exactly one fold
    union = np.zeros(V, dtype=bool)
    for m in masks:
        assert not np.any(union & m), "node assigned to multiple folds"
        union |= m
    assert np.all(union), "not all nodes assigned to a fold"

    # Sizes differ by at most 1
    sizes = [int(m.sum()) for m in masks]
    assert max(sizes) - min(sizes) <= 1


def test_fold_split_mask_skip_when_folds_zero():
    from src.models.inference import _cv_fold_mask
    # cv_folds=0 is handled upstream; _cv_fold_mask requires cv_folds >= 1
    with pytest.raises(ValueError):
        _cv_fold_mask(10, 0, 0)
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
uv run pytest test/test_cv_pipeline.py::test_fold_split_mask -v
```
Expected: `ImportError` — `_cv_fold_mask` not defined.

- [ ] **Step 3: Add `_cv_fold_mask` to `src/models/inference.py`**

Add before `run_stan_inference`:

```python
def _cv_fold_mask(V: int, cv_folds: int, cv_fold: int) -> np.ndarray:
    if cv_folds < 1:
        raise ValueError(f"cv_folds must be >= 1, got {cv_folds}")
    return (np.arange(V) % cv_folds) == cv_fold
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
uv run pytest test/test_cv_pipeline.py::test_fold_split_mask test/test_cv_pipeline.py::test_fold_split_mask_skip_when_folds_zero -v
```
Expected: 2 tests PASS.

- [ ] **Step 5: Add CV split and output write to `run_stan_inference`**

In `run_stan_inference`, locate the block where `V = M.shape[0]` is set for both the twitter and OO branches (around line 963 / 997). After `V` is set and before `stan_file_path` is resolved, insert:

```python
    # --- CV fold splitting ---
    cv_folds = int(config.get('cv_folds', 0))
    cv_fold  = int(config['model'].get('cv_fold', config.get('cv_fold', -1)))
    run_cv = cv_folds > 0 and cv_fold >= 0

    M_held = None
    N_held = None

    if run_cv:
        if not requires_2d:
            print(
                f"Warning: cv_folds={cv_folds} requested but model '{stan_file}' uses "
                "aggregated (1D) data — CV requires individual node rows. Skipping CV."
            )
            run_cv = False
        else:
            held_mask = _cv_fold_mask(V, cv_folds, cv_fold)
            train_mask = ~held_mask
            M_held = M[held_mask]
            N_held = N[held_mask]
            M = M[train_mask]
            N = N[train_mask]
            if row_weights is not None:
                row_weights = row_weights[train_mask]
            V = int(M.shape[0])
            print(
                f"CV fold {cv_fold}/{cv_folds}: "
                f"train={V} nodes, held_out={int(held_mask.sum())} nodes"
            )
```

Then, after `fit = model.sample(...)` (and after the existing `fit.summary()` / `model_comparison` / kernel / posterior writes), append:

```python
    if run_cv and M_held is not None and M_held.shape[0] > 0:
        try:
            cv_result = compute_held_out_loglik(M_held, N_held, fit)
            cv_df = pd.DataFrame([{
                "cv_score":       cv_result["cv_score"],
                "n_held_out":     cv_result["n_held_out"],
                "mean_log_pred":  cv_result["mean_log_pred"],
                "cv_fold":        cv_fold,
                "cv_folds":       cv_folds,
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
        except Exception as e:
            print(f"Warning: CV evaluation failed ({e}). Skipping cv_held_out_loglik.parquet.")
```

- [ ] **Step 6: Run existing tests to check nothing regressed**

```bash
uv run pytest test/ -v
```
Expected: all pre-existing tests PASS; new fold-mask tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/models/inference.py test/test_cv_pipeline.py
git commit -m "feat: add CV fold splitting to run_stan_inference"
```

---

## Task 3: `aggregate_cv.py` script

**Files:**
- Create: `workflow/scripts/analysis/aggregate_cv.py`
- Modify: `test/test_cv_pipeline.py`

- [ ] **Step 1: Write the failing test**

Append to `test/test_cv_pipeline.py`:

```python
def test_aggregate_cv_folds(tmp_path):
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(
        "aggregate_cv",
        "workflow/scripts/analysis/aggregate_cv.py",
    )
    agg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agg)

    # Write three fake fold parquet files
    fold_dirs = []
    for fold in range(3):
        d = tmp_path / f"fold_{fold}"
        d.mkdir()
        pd.DataFrame([{
            "cv_score": float(fold),
            "n_held_out": 10,
            "mean_log_pred": float(fold) / 10,
            "cv_fold": fold,
            "cv_folds": 3,
        }]).to_parquet(d / "cv_held_out_loglik.parquet", index=False)
        fold_dirs.append(str(d))

    out_path = tmp_path / "cv_summary.parquet"
    agg.aggregate_cv_folds(fold_dirs, str(out_path))

    result = pd.read_parquet(out_path)
    assert len(result) == 1
    assert abs(result["total_cv_lppd"].iloc[0] - (0.0 + 1.0 + 2.0)) < 1e-9
    assert result["n_total_nodes"].iloc[0] == 30
    assert abs(result["mean_cv_lppd_per_node"].iloc[0] - 3.0 / 30) < 1e-9
    assert result["cv_folds"].iloc[0] == 3
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
uv run pytest test/test_cv_pipeline.py::test_aggregate_cv_folds -v
```
Expected: `FileNotFoundError` — `aggregate_cv.py` does not exist.

- [ ] **Step 3: Create `workflow/scripts/analysis/aggregate_cv.py`**

```python
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def aggregate_cv_folds(fold_dirs: list[str], output_path: str) -> pd.DataFrame:
    records = []
    for d in fold_dirs:
        p = Path(d) / "cv_held_out_loglik.parquet"
        if not p.exists():
            print(f"Warning: {p} not found — skipping this fold.")
            continue
        records.append(pd.read_parquet(p))

    if not records:
        raise ValueError(f"No cv_held_out_loglik.parquet files found in {fold_dirs}")

    df = pd.concat(records, ignore_index=True)
    total_nodes = int(df["n_held_out"].sum())
    total_lppd = float(df["cv_score"].sum())

    summary = pd.DataFrame([{
        "cv_folds":             int(df["cv_folds"].iloc[0]),
        "total_cv_lppd":        total_lppd,
        "mean_cv_lppd_per_node": total_lppd / total_nodes if total_nodes > 0 else float("nan"),
        "n_total_nodes":        total_nodes,
    }])
    summary.to_parquet(output_path, index=False)
    print(
        f"CV aggregate: total_cv_lppd={total_lppd:.4f}  "
        f"mean_per_node={summary['mean_cv_lppd_per_node'].iloc[0]:.4f}  "
        f"n_nodes={total_nodes}"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-dirs", nargs="+", required=True,
                        help="Inference output directories for each CV fold")
    parser.add_argument("--output", required=True,
                        help="Path for cv_summary.parquet")
    args = parser.parse_args()
    aggregate_cv_folds(args.fold_dirs, args.output)
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
uv run pytest test/test_cv_pipeline.py::test_aggregate_cv_folds -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add workflow/scripts/analysis/aggregate_cv.py test/test_cv_pipeline.py
git commit -m "feat: add aggregate_cv.py fold aggregation script"
```

---

## Task 4: Snakemake integration

**Files:**
- Modify: `workflow/lib/experiment_plan.py`
- Create: `workflow/rules/cv.smk`
- Modify: `Snakefile`

- [ ] **Step 1: Add `cv_aggregate_outputs` to `WorkflowContext` in `experiment_plan.py`**

In `experiment_plan.py`, locate the `WorkflowContext` dataclass and add one field:

```python
@dataclass(frozen=True)
class WorkflowContext:
    # ... existing fields ...
    cv_aggregate_outputs: tuple[str, ...]   # add this
```

In `build_workflow_context` (or wherever `WorkflowContext` is constructed), compute `cv_aggregate_outputs`. The key idea: for each unique set of sweep params *excluding* `cv_fold`, compute the expected `cv_summary.parquet` path. The path uses the same `results/{cfg_name}/{source}/{data_name}/{group_id}/` structure where `group_id` encodes the non-fold params.

Find the line in `build_workflow_context` where `inference_outputs` is assembled. Near it, add:

```python
cv_folds = int(full_config.get("cv_folds", 0))
cv_aggregate_paths: list[str] = []

if cv_folds > 0:
    # Group sweep rows by all params except cv_fold.
    seen_groups: set[str] = set()
    for data_name, row in zip(data_names_for_rows, sweep_rows):
        group_row = {k: v for k, v in row.items() if k != "cv_fold"}
        group_instance_id = encode_instance_id(get_instance_dir(group_row))
        group_key = f"{data_name}/{group_instance_id}"
        if group_key not in seen_groups:
            seen_groups.add(group_key)
            cv_aggregate_paths.append(
                f"results/{cfg_name}/{source}/{data_name}/{group_instance_id}/cv_summary.parquet"
            )
```

Pass `cv_aggregate_outputs=tuple(cv_aggregate_paths)` into the `WorkflowContext(...)` constructor.

> **Note:** `data_names_for_rows` is the parallel list of data_name strings corresponding to each sweep row. Inspect how `inference_outputs` is built — it follows the same `(data_name, row)` pairing — and follow that exact pattern here.

- [ ] **Step 2: Add `cv_aggregate` target and include to `Snakefile`**

In `Snakefile`, after `PPC_OUTPUTS = ...`:

```python
CV_AGGREGATE_OUTPUTS = list(WORKFLOW_CONTEXT.cv_aggregate_outputs)
```

After `rule ppc:`:

```python
rule cv_aggregate:
    """Aggregate k-fold CV scores per model config."""
    input:
        CV_AGGREGATE_OUTPUTS
```

At the bottom, add:

```python
include: "workflow/rules/cv.smk"
```

- [ ] **Step 3: Create `workflow/rules/cv.smk`**

```python
import json
import re


def _cv_fold_dirs(wildcards):
    """Return inference output dirs for all folds of a given group."""
    cv_folds = int(config.get("cv_folds", 0))
    group_id = wildcards.group_instance_id  # e.g. "K~2__stan_file~non_parametric_monotonic"
    # Re-insert cv_fold~N for each fold to reconstruct per-fold instance_id
    fold_dirs = []
    for fold in range(cv_folds):
        fold_part = f"cv_fold~{fold}"
        # Insert cv_fold into the sorted key sequence.
        parts = group_id.split("__")
        parts_with_fold = sorted(parts + [fold_part])
        fold_instance_id = "__".join(parts_with_fold)
        fold_dirs.append(
            f"results/{wildcards.sweep_name}/{wildcards.data_source}/"
            f"{wildcards.data_name}/{fold_instance_id}"
        )
    return fold_dirs


rule run_cv_aggregate:
    input:
        fold_dirs = _cv_fold_dirs,
        script    = "workflow/scripts/analysis/aggregate_cv.py",
    output:
        "results/{sweep_name}/{data_source}/{data_name}/{group_instance_id}/cv_summary.parquet"
    run:
        import subprocess, sys
        fold_dir_args = " ".join(f'"{d}"' for d in input.fold_dirs)
        cmd = (
            f"uv run python {input.script} "
            f"--fold-dirs {fold_dir_args} "
            f"--output {output[0]}"
        )
        shell(cmd)
```

- [ ] **Step 4: Smoke-test the DAG with the example synthetic config**

Create a minimal test config at `configs/cv_smoke_test.yaml` (adapt from `configs/example_synthetic_workflow.yaml`):

```yaml
name: cv_smoke_test
source: synthetic

synthetic:
  dataset_name: cv_smoke
  n_nodes: 100
  kmax: 5
  # ... match whatever example_synthetic_workflow.yaml uses for synthetic generation

cv_folds: 3

model:
  stan_file: src/models/non_parametric_monotonic.stan
  K: 1
  sampling:
    chains: 1
    iter_warmup: 100
    iter_sampling: 100

sweep:
  K: [1, 2]
  cv_fold: [0, 1, 2]
```

Run a dry-run to verify the DAG is correct (no actual jobs submitted):

```bash
uv run snakemake cv_aggregate --configfile configs/cv_smoke_test.yaml \
  --profile workflow/profiles/local -n
```
Expected: Snakemake prints a job summary showing 6 inference jobs (2 K × 3 folds) + 2 aggregate jobs (one per K). No errors.

- [ ] **Step 5: Run the smoke test end-to-end**

```bash
just sweep configs/cv_smoke_test.yaml
uv run snakemake cv_aggregate --configfile configs/cv_smoke_test.yaml \
  --profile workflow/profiles/local
```
Expected:
- `results/cv_smoke_test/synthetic/cv_smoke/{instance_id}/cv_held_out_loglik.parquet` exists for all 6 fold jobs
- `results/cv_smoke_test/synthetic/cv_smoke/{group_id}/cv_summary.parquet` exists for both K=1 and K=2
- K=2 `total_cv_lppd` ≥ K=1 (not a hard requirement — just verify the scores exist and differ)

- [ ] **Step 6: Commit**

```bash
git add workflow/lib/experiment_plan.py workflow/rules/cv.smk Snakefile configs/cv_smoke_test.yaml
git commit -m "feat: add Snakemake cv_aggregate target and cv.smk rule"
```

---

## Self-Review Notes

**Spec coverage:**
- Section 2 (Config) → Task 4 steps 1-2 ✓
- Section 3 (Data splitting) → Task 2 step 5 ✓
- Section 4 (Held-out lppd) → Task 1 step 3 ✓
- Section 5 (Aggregation) → Task 3 + Task 4 step 3 ✓
- Edge case: 1D models skip CV with warning → Task 2 step 5 ✓
- Edge case: `cv_folds=0` no-op → covered by `run_cv = cv_folds > 0 and cv_fold >= 0` guard ✓

**Type consistency:**
- `compute_held_out_loglik` returns dict with keys `cv_score`, `n_held_out`, `mean_log_pred` — used consistently in Task 2 step 5 and Task 3
- `_cv_fold_mask` returns `np.ndarray[bool]` — used correctly in Task 2 step 5
- `aggregate_cv_folds(fold_dirs: list[str], output_path: str)` signature matches Task 3 test

**One known ambiguity in Task 4 Step 1:** The exact variable names for `data_names_for_rows` and `sweep_rows` pairing inside `build_workflow_context` depend on the current `experiment_plan.py` internals. Read `workflow/lib/experiment_plan.py` in full before implementing Step 1 and follow the same loop structure used to build `inference_outputs`.
