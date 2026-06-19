# het-ker

Simulation + inference experiments for heterogeneous contagion kernels.

## Layers

| Path | Role |
|---|---|
| `src/` | Library: simulation, inference (`fit_stan`, CV), analysis |
| `configs/models/` | Stan model default configs (`nonparametric_eb.yaml`, `simple.yaml`) |
| `workflow/` | Orchestration: build experiment manifests, simulate DGPs, run tasks |
| `manifests/` | Generated JSONL job lists (one line = one task) |
| `results/` | Outputs: `data_cache/` for shared simulated data, `<experiment>/` for task outputs |

## Running a new experiment

**Edit only `workflow/build_manifest_inference.py`** — `run_task.py` and `dgp.py` stay fixed.

1. Bump `EXPERIMENT` (controls the output directory name and manifest filename).
2. Edit the swept parameter lists.
3. Edit `MODELS` to change which models are fit (each entry = one more task per DGP).
4. Edit the `dgp` dict inside `main()` if the DGP structure itself changes.

## Run loop

```bash
# 1. Build manifest — prints the array range to use in the next step
uv run python workflow/build_manifest_inference.py

# 2. Submit — set --array from the printed range, e.g. --array=0-1799
#    (submit_*.sh calls: python run_task.py manifests/<exp>.jsonl $SLURM_ARRAY_TASK_ID results/)
sbatch --array=0-1799 submit_inference.sh

# 3. Aggregate (once jobs finish)
uv run python workflow/aggregate_inference.py results/inference_epsilon_omega_sweep
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
