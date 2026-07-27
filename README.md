<p align="left">

<a href="https://github.com/kaiser-dan/heterogeneous-infection-kernels/blob/main/LICENSE" target="_blank">
<img alt="License: BSD-3" src="https://img.shields.io/github/license/kaiser-dan/heterogeneous-infection-kernels">
</a>

<a href="https://www.python.org/" target="_blank">
<img alt="Made with Python" src="https://img.shields.io/badge/made%20with-python-1f425f.svg">
</a>

<a href="https://doi.org/10.5281/zenodo.21632560">
<img src="https://zenodo.org/badge/1070949785.svg" alt="DOI">
</a>

<a href="https://arxiv.org/abs/ARXIVID" target="_blank">
<img alt="arXiv: ARXIVID" src="https://img.shields.io/badge/arXiv-ARXIVID-red.svg">
</a>

</p>

# Emergent contagion complexity: Disentagling mechanistic complexity from correlated heterogeneity

The repository contains code accompanying the following paper (in preparation): Tang, K. _et al_. 2026. [Emergent contagion complexity: Disentagling mechanistic complexity from correlated heterogeneity](https://arxiv.org/abs/ARXIVID).

If you use this repository in any fashion in your own work, please cite our work according to the `CITATION.cff`.

# Contents

- [Contents](#contents)
- [Getting Started](#getting-started)
  - [Repository layout](#repository-layout)
  - [Installing](#installing)
- [Usage](#usage)
  - [Using `src/` directly](#using-src-directly)
  - [Running the pipeline](#running-the-pipeline)
  - [Gotchas](#gotchas)
- [Other Information](#other-information)
  - [Built With](#built-with)
  - [Authors](#authors)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)

# Getting started

There are two ways to use this repo:

- **As a library** (`src/`) — simulate a data-generating process (DGP) and fit a model in a notebook or script, no
  Snakemake required.
- **As a pipeline** (`workflow/`, driven by `config/`) — run the full set of experiments
  (parameter sweeps, cross-validation, posterior-predictive check) via Snakemake, with results
  cached and resumed automatically.

## Repository layout

| Path | Role |
|---|---|
| `src/simulation/` | Build a graph, a bank of kernels; assign kernels to nodes; run the contagion process; extract activation counts |
| `src/inference/` | Fit the Stan mixture model (`fit_stan`) and run cross-validation folds (`run_cv`) |
| `src/analysis/` | Complexity scoring and posterior-predictive p-values |
| `src/models/` | Stan model source (`.stan`) |
| `config/config.yaml` | Snakemake config file (currently just `results_root`) |
| `config/gen_params_*.py` | One generator per experiment; each defines that experiment's full parameter grid |
| `config/gen_all_params.py` | Runs every generator, writing `config/params/*.tsv` |
| `config/params/*.tsv` | Committed, generated parameter frames — the actual experiment definitions that Snakemake reads |
| `config/models/*.yaml` | Default Stan model hyperparameters (`nonparametric_eb.yaml`, `simple.yaml`) |
| `workflow/Snakefile` | Top-level orchestration: includes the rule files, defines `rule all` |
| `workflow/rules/*.smk` | Rule definitions for the three experiment families (sweeps, CV, PPC) |
| `workflow/scripts/` | The `script:` bodies each rule runs, plus shared helpers (`params_io.py`, `dgp_shapes.py`) |
| `workflow/dgp.py` | `simulate_MN`: the single function mapping a DGP dict to activation counts `(M, N)` |
| `bin/compile_models.py` | Compiles every Stan model in `src/models/` |
| `data/` | Static inputs: an example edgelist graph, community labels, a nonparametric kernel |
| `results/` | Pipeline outputs |

## Installation

```bash
uv sync --all-extras --all-groups   # full install (dev extras included)
uv sync --no-dev                    # minimal install
```

Stan models are compiled on demand by `cmdstanpy`, but to precompile all of them:

```bash
just compile          # == uv run bin/compile_models.py
```

CmdStan itself must already be installed (`uv run python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"`
once, if you've never used CmdStanPy before).

## Using `src/` directly

Nothing in `src/` depends on Snakemake; `src/` is a plain importable package, so from the repo root 
(or with the repo root on `sys.path`, e.g., run `uv run jupyter` from there) you can import it straight into 
a notebook:

```python
from src.simulation import (
    build_graph, build_kernel_bank, assign_kernels,
    build_initial_state, heterogeneous_contagion_process, get_activations,
)
from src.inference import fit_stan, get_posterior_draws
from src.analysis import compute_complexity_scores, simple_reference_logit

# 1. Build a graph and a bank of per-node kernels
adj, kmax = build_graph({"type": "sbm", "params": {"n": 256, "avg_degree": 18, "epsilon": 0.9}}, seed=0)
bank = build_kernel_bank(
    [{"type": "simple", "beta": 0.01}, {"type": "simple", "beta": 0.05}], kmax=kmax,
)
labels = assign_kernels(adj.shape[0], {"mode": "block_flip", "omega": 0.1}, n_kernels=2, seed=0)
c = np.stack([bank[labels[i]] for i in range(adj.shape[0])])

# 2. Run the contagion process and collect activation counts
x0 = build_initial_state(adj.shape[0], initial_infected=0.125, seed=0)
x = heterogeneous_contagion_process(adj, gamma=0.1, c=c, x0=x0, tmax=1000, random_seed=0)
M, N = get_activations(x.T, adj, kmax)

# 3. Fit the nonparametric mixture model and score its complexity
result = fit_stan(M, N, "config/models/nonparametric_eb.yaml", {"K": 1})
mu_draws = get_posterior_draws(result, "mu")[:, 0, :]          # (S, kmax), K=1
ref = simple_reference_logit(beta=0.01, kmax=kmax)             # simple-contagion reference
scores = compute_complexity_score(mu_draws, ref)               # {complexity, ...}
```

`workflow/dgp.py::simulate_MN` wraps steps 1–2 above into one call given a DGP dict (see next
section). Every Snakemake script uses it, and it's just as usable standalone:

```python
from workflow.dgp import simulate_MN

M, N = simulate_MN({
    "graph": {"type": "sbm", "params": {"n": 256, "avg_degree": 18, "epsilon": 0.9, "delta": 0.0}},
    "gamma": 0.1, "rho": 0.125, "tmax": 1000, "rep": 0,
    "kernels": [{"type": "simple", "beta": 0.01}, {"type": "simple", "beta": 0.05}],
    "assignment": {"mode": "block_flip", "omega": 0.1},
})
```

Every randomness source inside `simulate_MN` (graph draw, kernel-label assignment, initial
state, contagion trajectory) is seeded from `dgp["rep"]` for reproducibility.

For cross-validation on activation data you already have, `src.inference.cv.run_cv(M, N, n_folds, model_config_path)`
runs all folds and returns one `FoldResult` (fold-level and per-node held-out log predictive
density) per fold; `run_one_fold` runs a single fold, which is what the pipeline parallelizes over.

### DGP schema reference

A DGP dict has four top-level keys: `graph`, `kernels`, `assignment` (only meaningful when
`len(kernels) > 1`), and the scalars `gamma`, `rho`, `tmax`, `rep`.

**`graph`** — `{"type": ..., "params": {...}}`:

| `type` | Required params | Notes |
|---|---|---|
| `sbm` | `n`, `avg_degree`, `epsilon`, `B` | B-block SBM. `epsilon` ∈ (0,1): higher = stronger community structure. `B` (default 2). `delta` (default 0): asymmetry between block densities, only supported for `B=2`. |
| `er` | `n`, one of `avg_degree` or `p` | Erdős–Rényi. |
| `zkc` | _(none)_ | Zachary's karate club (fixed graph, 34 nodes). |
| `edgelist` | `filename` | Loads an undirected, unweighted edgelist from disk. |

**`kernels`** — a list of `{"type": ..., ...}` dicts, one per kernel mode. A one-element list
means homogeneous contagion (`assignment` is then ignored):

| `type` | Required params (in addition to `kmax`) | Notes |
|---|---|---|
| `simple` | `beta` | Independent cascade: c(ν) = 1 − (1−β)^ν |
| `sigmoid` | `rho_l`, `rho_h`, `w`, `k0` | Sigmoidal kernel |
| `power` | `lam`, `alpha` | c(ν) = 1 − exp(−λν^α); sublinear for α<1, superlinear for α>1 |
| `nonparametric` | `values` (or `values_file` in `workflow/dgp.py`) | Explicit list of probabilities, one per exposure level |

**`assignment`** — `{"mode": ..., ...}`, ignored unless `len(kernels) > 1`:

| `mode` | Required params | Notes |
|---|---|---|
| `block_flip` | `omega`, `communities` (optional) | Assigns kernels by graph block (or `communities` vector), then flips each node to another kernel with probability `omega` (uniform over the others). `omega = 0` is perfect alignment; `omega = 1/n_kernels` is uniform random. |
| `iid` | `probs` (optional) | Each node draws a kernel index independently; uniform if `probs` omitted. |
| `explicit` | `labels` | An `(n,)` integer array of kernel indices, passed directly. |

**Scalars**: `gamma` (recovery rate), `rho` (initial infected fraction or count), `tmax`
(simulation steps), `rep` (seed — deterministic given (dgp params, rep)).

### Model configs

`fit_stan(M, N, config_path, instance_params)` reads a YAML config, applies `instance_params`
as overrides (no need to edit the YAML for per-run changes), and fits the named Stan model:

| Config | Stan model | Key hyperparameters |
|---|---|---|
| `config/models/nonparametric_eb.yaml` | `mixture_nonparametric_eb.stan` | `K`, `logit_p_location`, `logit_p_scale`, `delta_concentration`, `sigma_scale`, `pi_concentration` |
| `config/models/simple.yaml` | `mixture_simple.stan` | `K`, `logit_beta_location`, `logit_beta_scale`, `sigma_scale`, `pi_concentration` |

`logit_*_location` accepts `"empirical"` / `"empirical_floor"` / `"empirical_ceiling"` in
addition to a fixed number. These resolve to data-derived logit-scale priors computed inside
`fit_stan` from the raw `(M, N)` exposure-level rates.

## Running the pipeline

The pipeline is three independent experiment families, each with its own rule file:

| Experiment | Rules (`workflow/rules/`) | What it produces |
|---|---|---|
| `beta_delta_sweep`, `epsilon_omega_sweep` | `sweep2d.smk`: `fit_model` → `measure_complexity` | Per-DGP and per-setting complexity scores across a 2D grid |
| `cv_basic` | `cv.smk`: `run_cv_task` → `aggregate_cv` | Cross-validated log predictive density and 1-SE-rule model selection across candidate `K` |
| `ppc` | `ppc.smk`: `ppc_observed` → `ppc_null` (×N) → `ppc_summary` | A posterior-predictive p-value for excess complexity, against a simple-contagion null |

### How an experiment's parameters flow to a job

Each experiment's full parameter grid is defined once, in a `config/gen_params_<name>.py`
generator and committed as `config/params/<name>.tsv` (plus a `<name>.wildcard_columns.json`
naming that frame's swept columns). Regenerate them with:

```bash
just params          # uv run python config/gen_all_params.py config/params
just check-params    # fails if the committed frames are stale relative to the generators
```

`workflow/scripts/params_io.py::load_experiment` loads a frame into a `snakemake.utils.Paramspace`
plus an `ExperimentFrame` that resolves a job's wildcards back to its one full row (fixed + swept
columns merged). `workflow/scripts/dgp_shapes.py` then turns that row into a DGP dict
(`build_sweep_sbm_dgp`, `build_cv_dgp`) for `simulate_MN`.

Rule output paths are literally the `Paramspace` instance patterns, e.g., one `beta_delta_sweep`
fit lands at `results/fits/beta_delta_sweep/model~np_rep~0_beta2~0.03_delta~-0.575.npz`, and one
CV fold at `results/cv/cv_basic/dgp_label~one_simple_K~1_rep~3_fold~2.parquet`.

### Adding or extending an experiment

1. Write (or extend) `config/gen_params_<name>.py`: it needs `EXP` (the name), `WILDCARD_COLUMNS`
   (the swept columns), and a `build_frame()` returning the full grid as a DataFrame — fixed
   columns identical on every row, swept columns varying, any nested structure (kernel lists,
   sampling configs) JSON-encoded via `gen_params_utils.json_col` into a `*_json` column.
2. Register the generator in `config/gen_all_params.py`'s `GENERATORS` list, then run `just params`.
3. If the row needs a new shape of DGP dict, add a builder to `workflow/scripts/dgp_shapes.py`.
4. Add or extend a `workflow/rules/*.smk` file: call `params_io.load_experiment(...)` to get an
   `ExperimentFrame`, then write rules whose `params:` pull from `EF.row(wc)` and whose
   `script:` does the actual simulate/fit/score work.
5. `include:` the new rule file from `workflow/Snakefile` and add its final output(s) to `rule all`.

### Output layout

```
results/
  fits/<sweep>/
    model~<model>_rep~<rep>_<axis1>~<v1>_<axis2>~<v2>.npz   # posterior draws (save_vars)
  complexity/<sweep>/
    complexity_per_dgp.parquet       # one row per (axis1, axis2, rep)
    complexity_by_setting.parquet    # averaged over reps per (axis1, axis2)
  cv/cv_basic/
    dgp_label~<label>_K~<k>_rep~<rep>_fold~<fold>.parquet   # one row per held-out node
    _cv_curve.parquet       # mean/SE lppd per (dgp_label, rep, K), pooled across folds
    _cv_selection.parquet   # K selected by 1-SE rule per (dgp_label, rep)
  ppc/
    observed.npz, nulls/r~<r>.npz, ppc_result.json, ppc_null_distribution.npz
```


# Other Information

## Built With

- [ChooseALicense](https://choosealicense.com/) - used to choose the license.
- [Snakemake](https://snakemake.readthedocs.io/en/stable/) - used for organizing and running research pipeline.
- [Stan](https://mc-stan.org/) - used for writing inference models.
- [cmdstanpy](https://mc-stan.org/cmdstanpy/) - used to interface to Stan through Python.

## Authors

All correspondence should be directed to [Daniel Kaiser](mailto:daniel.kaiser@virginia.edu) or [Katerina Tang](mailto:kbt28@cornell.edu).

- [Katerina Tang](https://kbtang28.github.io)
- [Daniel Kaiser](https://kaiser-dan.github.io)
- [William Thompson](https://www.willhwthompson.com/)
- [Jean-Gabriel Young](https://jgyoung.ca/)
- [Laurent Hébert-Dufresne](https://laurenthebertdufresne.github.io/)
- [Nicholas W. Landry](https://nwlandry.com/)

## License

This project is licensed under the [BSD-3 License](LICENSE) - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Billie Thompson** - *Provided original README* - [PurpleBooth](https://github.com/PurpleBooth)
- **George Datseris** - *Published workshop on scientific code; inspired organization for reproducibility* - [GoodScientificCodeWorkshop](https://github.com/JuliaDynamics/GoodScientificCodeWorkshop)
- **Nicolò Ruggeri** - *Inspired shield use in README* - [Nicolò Ruggeri's GitHub](https://github.com/nickruggeri)

