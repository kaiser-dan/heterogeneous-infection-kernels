ROOT := `pwd`

# ----------------------
# List commands
default:
    @just --list --unsorted

# ----- Setup & Install -----
# Install according to uv.lock, with all extras and optional dependencies
[group('setup')]
install:
    uv sync --all-extras --all-groups

# Minimum install according to uv.lock
[group('setup')]
installmin:
    uv sync --no-dev

# Run all unit tests
[group('setup')]
test:
    uv run pytest

# Compile all Stan models
[group('setup')]
compile:
    uv run bin/compile_models.py

# ----- Research -----
# Generate all processed data via Snakemake
[group('research')]
data:
    uv run snakemake -c1

# Totally clear and regenerate all processed data
[group('research')]
data-total:
    rm -rf data/processed
    uv run snakemake -c1

# Render all Quarto documents
[group('research')]
renderall:
    find ./ -type f -name "*.qmd" | xargs -I{} quarto render {}

# Run a research sweep defined in a YAML config
[group('research')]
sweep config profile="workflow/profiles/local":
    uv run python scripts/sweep.py {{config}} --profile {{profile}}


# ----- Sync -----
# Sync results from HPC to local
[group('sync')]
sync:
    rsync -avP vacc:/gpfs1/home/w/t/wthomps3/CSDS/side_projects/heterogeneous-infection-kernels/results/ results/


# ----- Cleanup -----
# Remove temporary and test files
[confirm]
[group('clean')]
clean:
    rm -rf results/
    find ./ -type f -name "*tmp" -delete

# Remove environment
[confirm]
[group('clean')]
rmenv:
    rm -rf .venv
