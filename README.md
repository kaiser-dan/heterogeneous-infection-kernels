# het-ker

Simulation + inference experiments for heterogeneous contagion kernels.

## Relevant directories

| Path | Role |
|---|---|
| `src/` | Library: simulation, inference (`fit_stan`, CV), analysis |
| `configs/models/` | Stan model default configs (`nonparametric_eb.yaml`, `simple.yaml`) |
| `workflow/` | Experiments: build manifests, simulate DGPs, run tasks |
| `manifests/` | Generated JSONL job lists (one line = one task) |
| `results/` | Outputs: `data_cache/` for shared simulated data, `<experiment>/` for task outputs |

## Running a new experiment

**Edit only the relevant manifest builder;** `run_task.py` and `dgp.py` stay fixed.

**`build_manifest_inference.py`** for parameter sweeps, fits each model on full data:
1. Bump `EXPERIMENT`.
2. Edit the swept parameter lists (`EPSILONS`, `OMEGAS`, `REPS`, `FIXED`, `GRAPH_BASE`).
3. Edit `MODELS`. Each entry adds one task per DGP; `save_vars` controls which posterior variables are written to the output `.npz`.
4. Edit the `dgp` dict in `main()` if the DGP structure changes.

**`build_manifest_cv.py`** for model selection via K-fold CV:
1. Bump `EXPERIMENT`.
2. Edit the fixed parameters (`FIXED`, `GRAPH`).
3. Edit `DGP_FAMILIES` (named ground-truth DGPs to evaluate against), `N_FOLDS`, and `REPS`.
4. Edit `CANDIDATES`. No `save_vars`; each entry adds one fold-task per DGP × rep.

## Run loop

Replace `<kind>` with `inference` or `cv`:

```bash
# 1. Build manifest — prints the array range to use in the next step
uv run python workflow/build_manifest_<kind>.py

# 2. Submit — set --array from the printed range, e.g. --array=0-1799
#    (submit_*.sh calls: python run_task.py manifests/<exp>.jsonl $SLURM_ARRAY_TASK_ID results/)
sbatch --array=0-1799 submit_<kind>.sh

# 3. Aggregate (once jobs finish)
uv run python workflow/aggregate_<kind>.py results/<experiment>
```

## What gets saved

```
results/
  data_cache/
    data_<hash>.npz           # shared (M, N) arrays; one file per unique DGP
    data_<hash>.json          # sidecar: full DGP dict keyed by the same hash
  <experiment>/
    <task_name>.npz           # posterior draws for inference tasks
    <task_name>_fold.parquet  # CV fold record
```

`dgp_key(dgp)` → MD5 hash (12 chars); identical params → identical file, shared across all models and folds.

`lookup_dgp(key)` resolves a hash back to its full DGP dict via the sidecar.

Aggregation writes combined parquet tables into `results/<experiment>/`.

## Configurable options

### Network (`graph` key in DGP)

| `type` | Required params | Notes |
|---|---|---|
| `sbm` | `n`, `avg_degree`, `epsilon` | Two-block SBM. `epsilon` ∈ (0,1): higher = stronger community structure. `delta` (default 0): asymmetry between block densities. |
| `er` | `n`, one of `avg_degree` or `p` | Erdős–Rényi. |
| `zkc` | _(none)_ | Zachary's karate club (fixed graph, 34 nodes). |

### Contagion kernel (`kernels` list in DGP)

Each entry in `kernels` is one kernel type; use a one-element list for homogeneous contagion.

| `type` | Required params | Notes |
|---|---|---|
| `simple` | `beta` | Independent cascade: P(ν) = 1 − (1−β)^ν |
| `hill` | `p`, `h`, `tau` | Sigmoidal saturation: P(ν) = pνʰ / (τʰ + νʰ) |
| `nonparametric` | `values` | Explicit list of probabilities, one per exposure level |

### Kernel assignment (`assignment` key in DGP)

Only used when `len(kernels) > 1`; ignored for single-kernel DGPs.

| `mode` | Required params | Notes |
|---|---|---|
| `block_flip` | `omega`, `communities` (optional) | Assigns kernels by SBM block (denoted by `communities` vec), then flips each node with probability `omega`. Requires exactly 2 kernels. |
| `iid` | `probs` (optional) | Each node draws a kernel index independently; uniform if `probs` omitted. |
| `explicit` | `labels` | Pass an (n,) integer array of kernel indices directly. |

### DGP scalars (typically set in `FIXED`)

| Key | Role |
|---|---|
| `gamma` | Recovery rate |
| `rho` | Initial infected fraction |
| `tmax` | Simulation time steps |
| `rep` | Random seed (graph, labels, trajectory all derived from this) |

### Models / Candidates (`MODELS` or `CANDIDATES` dict)

Both dicts share the same config structure; the only difference is that `MODELS` (inference) includes `save_vars` and `CANDIDATES` (CV) does not; CV tasks write fold records instead of draws.

```python
# inference (MODELS)
"np": {
    "config": "configs/models/nonparametric_eb.yaml",  # path to YAML
    "instance_params": {                               # overrides YAML defaults
        "K": 1,                                        # number of mixture components
        "sampling": {"iter_sampling": 500, ...},
    },
    "save_vars": ["mu"],   # posterior variables written to the output .npz
}

# CV (CANDIDATES) — identical except no save_vars
"np_K2": {
    "config": "configs/models/nonparametric_eb.yaml",
    "instance_params": {"K": 2, "sampling": {...}},
}
```

Available model configs (in `configs/models/`):

| File | Stan model | Key hyperparameters |
|---|---|---|
| `nonparametric_eb.yaml` | `mixture_nonparametric_eb.stan` | `K`, `logit_p_location`, `logit_p_scale`, `delta_concentration`, `sigma_scale`, `pi_concentration` |
| `simple.yaml` | `mixture_simple.stan` | `K`, `logit_beta_location`, `logit_beta_scale`, `sigma_scale`, `pi_concentration` |

`instance_params` in the manifest entry override the YAML defaults at runtime---no need to edit the YAML for per-sweep changes.

## Gotchas

- **DGP schema is shared** between `build_manifest_*.py` and `dgp.py`. If you rename or add DGP fields, rebuild the manifest *and* delete `results/data_cache/`. Stale cache files will silently reuse old data.
- The `dgp_key` hash covers the entire DGP dict. Changing even one field (e.g. `tmax`) produces a new hash and triggers fresh simulation.
- `run_task.py` skips tasks whose output already exists; safe to requeue failed SLURM array jobs without duplicating work.
- `rep` is used as the random seed throughout; same `rep` yields same graph, kernel labels, and trajectory.