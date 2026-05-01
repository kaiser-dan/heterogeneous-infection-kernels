# CV Pipeline Design: K Selection for Non-Parametric Mixture Models

**Date:** 2026-04-30  
**Status:** Approved  
**Goal:** Select the number of mixture components K for the non-parametric contagion model using held-out predictive log-likelihood (k-fold CV), without modifying the existing sweep infrastructure.

---

## 1. Scientific Context

The Stan models infer a population-level infection kernel `p(infect | nu infected neighbors)`. Nodes are treated as iid draws from this population. `M[i, nu]` and `N[i, nu]` are already temporally aggregated — there is no temporal structure at the inference layer. Node-level row splitting is the correct CV unit: it tests whether the population kernel inferred from a training set generalizes to held-out individuals.

The per-node latents `z_alpha[i,k]` exist only for training nodes. Held-out predictive likelihood therefore uses the **population-level kernels** `mu[k]` (already in `generated quantities`) and mixture weights `pi`, marginalizing over component assignment k. This is the right quantity for K selection — we are asking whether K population-level components explains new nodes, not whether per-node deviations help.

---

## 2. Config Interface

CV is controlled by two fields:

```yaml
cv_folds: 5        # 0 or absent = skip CV, pipeline behaves exactly as today

sweep:
  K: [1, 2, 3]
  cv_fold: [0, 1, 2, 3, 4]   # added when cv_folds > 0; swept over like any other param
```

`cv_folds` is a top-level config param. `cv_fold` is a sweep dimension. The existing cartesian product expansion produces `|K| × cv_folds` inference jobs automatically. When `cv_folds: 0`, `cv_fold` is absent from the sweep and the pipeline is unchanged.

---

## 3. Data Splitting

Inside `run_stan_inference`, when `cv_fold` is present in `instance_params`:

```python
held_out_mask = (np.arange(V) % cv_folds) == cv_fold
M_train, N_train = M[~held_out_mask], N[~held_out_mask]
M_held,  N_held  = M[held_out_mask],  N[held_out_mask]
```

Modulo assignment is deterministic (no seed required), balanced for large V, and consistent across all K values (same held-out nodes for K=1, K=2, K=3 when the raw M/N ordering is fixed). Stan fits on `M_train / N_train` only.

Row weights (`ROW_WEIGHT`) are split by the same mask.

---

## 4. Held-Out Predictive Log-Likelihood

After fitting, compute the log posterior predictive density (lppd) on held-out nodes using `mu` and `pi` posterior samples. For held-out node `i` and posterior draw `s`:

```
log_lik[i, s] = logsumexp_k [ log(pi_k[s]) + Σ_ν M_held[i,ν]·log(μ_k[s][ν]) + N_held[i,ν]·log(1−μ_k[s][ν]) ]
```

Then per-node predictive log-lik (marginalizing over draws):

```
log_pred[i] = logsumexp_s(log_lik[i, s]) − log(S)
```

CV score for this fold:

```
cv_score = Σ_i log_pred[i]
```

`logsumexp` is used throughout to avoid underflow.

Output written to `{output_dir}/cv_held_out_loglik.parquet` with columns:

| column | type | description |
|---|---|---|
| `cv_score` | float | sum of log_pred over held-out nodes |
| `n_held_out` | int | number of held-out nodes |
| `cv_fold` | int | fold index |
| `cv_folds` | int | total number of folds |
| `mean_log_pred` | float | cv_score / n_held_out |

This file is written in addition to (not instead of) the existing `model_comparison.parquet`, `posterior_samples.parquet`, etc.

---

## 5. Aggregation

A new script `workflow/scripts/analysis/aggregate_cv.py` collects `cv_held_out_loglik.parquet` files across all folds for a given `(dataset, model_config)` combination and writes one `cv_summary.parquet` per model config:

| column | type | description |
|---|---|---|
| `K` | int | number of mixture components |
| `cv_folds` | int | total folds used |
| `total_cv_lppd` | float | Σ_fold cv_score (the primary selection metric) |
| `mean_cv_lppd_per_node` | float | total_cv_lppd / total V |
| `n_total_nodes` | int | total nodes across folds |

A new Snakemake target `cv_aggregate` depends on all fold inference jobs for a given config and triggers this script. Since `cv_fold` is a sweep dimension, each fold writes to its own output directory (e.g. `results/<name>/<source>/<dataset>/K=2_cv_fold=0/`). The aggregation script discovers fold outputs via a glob pattern over the results directory, grouping by all sweep params except `cv_fold`. The primary output for K selection is **`total_cv_lppd` vs `K`** — higher is better.

---

## 6. Implementation Scope

### Files modified
- `src/models/inference.py` — add fold splitting logic and `compute_held_out_loglik` function; write `cv_held_out_loglik.parquet`

### Files added
- `workflow/scripts/analysis/aggregate_cv.py` — fold aggregation script
- `workflow/rules/cv.smk` — `cv_aggregate` Snakemake rule (included from root Snakefile)

### Files unchanged
- All Stan model files
- All data processing scripts
- Existing inference output files (`posterior_samples.parquet`, `model_comparison.parquet`, etc.)
- Existing Snakemake rules

---

## 7. Invariants and Edge Cases

- **`cv_folds: 0` or absent**: `run_stan_inference` behaves exactly as today. No held-out split, no `cv_held_out_loglik.parquet`.
- **Minimum fold size**: If `V % cv_folds != 0`, modulo assignment gives folds differing by at most 1 node — acceptable.
- **Aggregated (1D) models**: Simple contagion models that aggregate M/N to 1D vectors have no V dimension; CV is undefined for them. The function should skip CV and log a warning when `requires_2d` is False.
- **Row weights**: Held-out predictive log-lik is computed without IPW (held-out nodes are evaluated at face value); IPW only affects the training fit.
