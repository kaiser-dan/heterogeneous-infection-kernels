ROOT := `pwd`

# List commands
default:
    @just --list --unsorted

# ----- Setup & Install -----

# Install all dependencies including dev extras
[group('setup')]
install:
    uv sync --all-extras --all-groups

# Minimal install (no dev dependencies)
[group('setup')]
installmin:
    uv sync --no-dev

# Run all unit tests
[group('setup')]
test:
    uv run pytest

# Compile all Stan models in src/inference/
[group('setup')]
compile:
    uv run bin/compile_models.py

# ----- Research -----

# Generate all synthetic datasets via Snakemake
[group('research')]
data:
    uv run snakemake -c1

# Clear and regenerate all synthetic datasets
[confirm]
[group('research')]
data-total:
    rm -rf data/synthetic
    uv run snakemake -c1

# Run inference sweep defined in a YAML config
[group('research')]
sweep config profile="workflow/profiles/local" *extra="":
    uv run snakemake inference --configfile {{config}} --profile {{profile}} {{extra}}

# Aggregate CV results for a sweep config
[group('research')]
cv config profile="workflow/profiles/local" *extra="":
    uv run snakemake cv_aggregate --configfile {{config}} --profile {{profile}} {{extra}}

# Compute complexity scores for a sweep config
[group('research')]
complexity config profile="workflow/profiles/local" *extra="":
    uv run snakemake complexity --configfile {{config}} --profile {{profile}} {{extra}}

# Run full pipeline: inference + CV + complexity
[group('research')]
pipeline config profile="workflow/profiles/local" *extra="":
    just sweep {{config}} {{profile}} {{extra}}
    just cv {{config}} {{profile}} {{extra}}
    just complexity {{config}} {{profile}} {{extra}}

# ----- Sync -----

# Sync results from HPC to local (excludes large posterior samples)
[group('sync')]
sync remote:
    rsync -avP --exclude='posterior_samples.parquet' {{remote}}results/ results/

# ----- Cleanup -----

# Remove all results
[confirm]
[group('clean')]
clean:
    rm -rf results/
    find ./ -type f -name "*tmp" -delete

# Remove synthetic data
[confirm]
[group('clean')]
clean-data:
    rm -rf data/synthetic

# Remove virtual environment
[confirm]
[group('clean')]
rmenv:
    rm -rf .venv
