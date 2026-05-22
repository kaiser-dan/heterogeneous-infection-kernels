# Heterogeneous infection kernels

Research project for HETKER.

The repository contains the original scientific analyses developed by the Authors (see below) for the paper

- **(In preparation)** LastName, FirstName. Year. [Paper title](arxiv.org).

If you use this repository in any fashion in your own work, please cite our work according to the `CITATION.cff`.

# Contents

- [Project Template](#project-template)
- [Contents](#contents)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installing](#installing)
  - [Quick Start](#quick-start)
- [Usage](#usage)
  - [Reproducing experiments](#reproducing-experiments)
  - [Applying to new data](#applying-to-new-data)
- [Documentation](#documentation)
- [Tests](#tests)
- [Other Information](#other-information)
  - [Built With](#built-with)
  - [Versioning](#versioning)
  - [Authors](#authors)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)

# Getting Started

The code base for this project is written in Python with package management handled with [`uv`](https://docs.astral.sh/uv/).

These instructions will give you a copy of the project up and running on
your local machine for development, testing, and analysis purposes.

## Prerequisites

A compatible Python install is needed to begin - the package management is handled by [`uv`](https://docs.astral.sh/uv/) as described below.

See the [`uv` installation instructions](https://docs.astral.sh/uv/getting-started/installation/) and make sure you have a working installation on your system before proceeding.

Optionally, if you prefer command runners in the GNUMake style, we've included a simple [`justfile`](https://github.com/casey/just).

## Installing

To (locally) reproduce this project, do the following:

1. Clone this repository. Notice that raw data are typically not included in the git-history and may need to be downloaded independently - see [Reproducing Experiments](#reproducing-experiments) for more information.
2. Create a virtual environment with the necessary packages. Using `uv`, this can be done with the single command run from the root directory:

```bash
just install
# or
uv sync --all-extras --all-groups
```

3. (Optional) If you want to reproduce the experiments using our original [`snakemake`](https://snakemake.readthedocs.io/en/stable/) workflow, install additional dependencies with the command:

```bash
just install
```

This will install all necessary packages and place them into a virtual environment, by default located at `./.venv/`.
To use the install package, either prefix all commands with `uv run` (preferred) - e.g., `uv run python --version` - or first activate the environment with `source .venv/bin/activate` (if on Unix-based system, `source .venv/Scripts/activate` if on Windows).

## Quick Start

The Jupyter notebook [`example.ipynb`](example.ipynb) contains a short working example and some comments to illustrate the basic usage of the developed method.

If you have followed the installation instructions above, you can open the notebook with the following command executed from the repository root:

```bash
uv run --with jupyter jupyter lab example.ipynb
```

This will open a Jupyter lab instance in your default browser.

# Usage

## Reproducing experiments

### Data Sources

We work with two primary data sources: 

- **Raw Data Downloads**: Place raw data in `data/raw/`. 

  - The data processing for the Twitter data was adapted from code provided by Bjarke Monstead for the original paper. DOI: 10.1371/journal.pone.0184148. **Note**: We filter for "active users" who have performed at least one retweet in our logs to ensure a valid contagion signal.
  - The data processing code for Operation Outbreak was adapted from code by Nate Blanchard.

- **Operation Outbreak**: Contact histories and infection logs from simulated outbreaks. Data is subset by simulation ID. The ten largest simulations are kept (e.g., sim 168 with 1200 participants).
  - The raw histories file is processed to extract contact networks and infection events.
  - Data produces: 
   - `sim_{id}_exposures.parquet`: Exposure counts for individuals.
   - `sim_{id}_edges.parquet`: Network topology.
   - `sim_{id}_M.parquet` / `sim_{id}_N.parquet`: Discretized incidence and survival matrices for Stan.

- **Twitter**: Retweet cascades and follower networks.
    - Partitioned by **intervention** (hashtag).
    - Matrices produced:
      - `A_matrix.parquet`: Exposure counts. $A^k_{i,j}$ = exposures of user $i$ to intervention $k$ via bot $j$.
      - `B_vector.parquet`: Adoption labels. $B^k_i = 1$ if user $i$ adopted.
      - `M_<k/kappa>.parquet`: Incidence matrix. $M^k_{i,m} = 1$ if adoption occurred at exposure level $m$.
      - `N_<k/kappa>.parquet`: Survival matrix. $N^k_{i,m} = 1$ if user survived exposure level $m$.

### Generating Processed Data

If you have the raw data files in `data/`, you can regenerate all processed matrices using the following commands:

- **Update data**: Only regenerates missing or modified files.
  ```bash
  just data
  ```

- **Full regeneration**: Deletes all processed data and rebuilds everything from scratch.
  ```bash
  just data-total
  ```

### Running the Snakemake Workflow

The project workflow is driven by the root [Snakefile](Snakefile). In practice you will usually use one of two entrypoints:

- `just sweep <config>`: the standard local wrapper for inference runs
- `uv run snakemake <target> --configfile <config> --profile <profile>`: the raw Snakemake command when you want full control

The `justfile` recipes are intentionally thin wrappers around the underlying Snakemake commands:

```bash
# Process the default raw-data targets declared in rule all
just data

# Equivalent raw Snakemake command
uv run snakemake -c1

# Run the inference target for a config with the local profile
just sweep configs/research_sweep_twitter.yaml

# Equivalent raw Snakemake command
uv run snakemake inference --configfile configs/research_sweep_twitter.yaml --profile workflow/profiles/local
```

The main Snakemake targets are:

- `all`: prepares the default processed-data targets from raw data
- `inference`: runs the configured preprocessing and inference outputs for a config file
- `ppc`: runs posterior predictive checks when `analysis.ppc.enabled: true`

The bundled profiles are:

- `workflow/profiles/local`: local execution on the current machine
- `workflow/profiles/slurm`: cluster execution through SLURM

Use the local profile for development and smoke tests. Use the SLURM profile when you want Snakemake to submit rules as cluster jobs.

### How the Config Files Work

Workflow configs are YAML files under `configs/`. The important top-level fields are:

- `name`: result namespace. Outputs are written under `results/<name>/...`
- `source`: one of `twitter`, `oo`, or `synthetic`
- `synthetic`, `twitter`, or `oo`: source-specific data-generation or data-selection settings
- `model`: default model and sampling settings passed into inference
- `sweep`: cartesian product of inference settings. Each row becomes one inference output directory
- `sim_sweep`: synthetic-only cartesian product over dotted keys under `synthetic`. Each row becomes one generated synthetic dataset before inference
- `analysis.ppc`: optional posterior predictive check settings

For synthetic workflows, the separation is:

- `synthetic`: how to generate one dataset
- `sim_sweep`: optional expansion into many synthetic datasets
- `sweep`: which model settings to fit to each generated dataset

### Synthetic Example

The file [configs/example_synthetic_workflow.yaml](configs/example_synthetic_workflow.yaml) is a minimal synthetic smoke-test config. It does three things:

1. generates one small SBM-based synthetic dataset
2. builds the synthetic summary directory that inference expects
3. fits one simple-contagion Stan model to that dataset

Inspect the DAG first:

```bash
uv run snakemake -n inference \
  --configfile configs/example_synthetic_workflow.yaml \
  --profile workflow/profiles/local
```

Run it locally through the `just` wrapper:

```bash
just sweep configs/example_synthetic_workflow.yaml
```

Run the equivalent raw Snakemake command directly:

```bash
uv run snakemake inference \
  --configfile configs/example_synthetic_workflow.yaml \
  --profile workflow/profiles/local
```

That example writes:

- processed synthetic data to `data/processed/synthetic/readme_synthetic_smoke/`
- the inference-ready summary to `data/summaries/synthetic/readme_synthetic_smoke.parquet/`
- posterior outputs to `results/example_synthetic_workflow/synthetic/readme_synthetic_smoke/base/`

Larger research runs use the same command shape with a different config, for example:

```bash
just sweep configs/research_sweep_twitter.yaml
just sweep configs/research_sweep_oo.yaml
just sweep configs/research_sweep_synthetic.yaml
just sweep configs/research_sim_sweep_synthetic.yaml
```

Results (posterior samples and summaries) are saved to the `results/` directory.

**NOTE**: I can provide the raw data. Unzip the `raw.zip` file in the `data/` directory. 

## Applying to new data

1. Prepare your data in a compatible format (see `src/data_processing/`).
2. Define a model sweep in a YAML configuration file.
3. Run the sweep using `just sweep your_config.yaml`.

# Documentation

This repository does not maintain extensive independent documentation for its source code. Functions have docstrings where needed and the [`example.ipynb`](example.ipynb) file can be used as a user guide.

# Tests

All unit tests are written with [pytest](docs.pytest.org).

Tests can be run directly with the commands:

```bash
just test
# or
uv run pytest
```

# Other Information

## Built With

### Repository and Version Control

- [ChooseALicense](https://choosealicense.com/) - Used to choose the license.
- [Commitizen]() - Used to maintain Conventional Commit standard.

### Source code

- [BeautifulSoup]() - Used for retrieving arXiv tags.

### Research analysis

- [Snakemake]() - Used for organizing and running research pipeline.
- [Quarto]() - Used for writing analyses documents.

## Versioning

We use [Semantic Versioning](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/kaiser-dan/heterogeneous-infection-kernels/tags) and our [published releases](https://github.com/kaiser-dan/heterogeneous-infection-kernels/releases).

> This repository is primarily intended to promote reproducibility and extensibility of a specific set of scientific analyses.
> There will likely be very few released versions, and the current release *should always be preferred if you want to apply the methodology to your own data*!

## Authors

All correspondence should be directed to [CORRESPONDING AUTHOR](mailto:email@iu.edu).

- AUTHOR
- AUTHOR
- AUTHOR

## License

This project is licensed under the [MIT License](LICENSE.md) Creative Commons License - see the [LICENSE](LICENSE.md) file for details.

## Acknowledgments

- **Billie Thompson** - *Provided original README* - [PurpleBooth](https://github.com/PurpleBooth)
- **George Datseris** - *Published workshop on scientific code; inspired organization for reproducibility* - [GoodScientificCodeWorkshop](https://github.com/JuliaDynamics/GoodScientificCodeWorkshop)
