ROOT := `pwd`

# ----------------------
# List commands
default:
    @just --list --unsorted

# ----- Setup & Install -----
[group('setup')]
install:
    uv sync --all-extras --all-groups

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
    uv run snakemake inference --configfile {{config}} --profile {{profile}}


# ----- Sync -----
# Sync results from HPC to local
[group('sync')]
sync:
    rsync -avP vacc:/gpfs1/home/w/t/wthomps3/CSDS/side_projects/heterogeneous-infection-kernels/results/ results/


# ----- Cleanup -----
# Remove temporary and test files
=======
# ----- Params -----

# Regenerate every experiment's committed Paramspace frame (config/params/*.tsv)
[group('params')]
params:
    uv run python config/gen_all_params.py config/params

# Fail if committed frames are stale relative to their generators
[group('params')]
check-params:
    rm -rf .tmp_check_params
    uv run python config/gen_all_params.py .tmp_check_params
    diff -rq config/params .tmp_check_params \
        && echo "config/params is up to date" \
        || (echo "STALE: config/params does not match the generators -- run 'just params' and commit the diff" && rm -rf .tmp_check_params && exit 1)
    rm -rf .tmp_check_params

# ----- Research -----
# All research targets run the Snakemake pipeline (workflow/Snakefile,
# config/params). 

# Show the DAG without running anything
[group('research')]
dryrun *extra="":
    uv run snakemake --snakefile workflow/Snakefile -n {{extra}}

# Run the full pipeline (all sweeps + cv + ppc)
[group('research')]
all cores="1":
    uv run snakemake --snakefile workflow/Snakefile --cores {{cores}}

# Run one numeric sweep through to its measure_complexity output
[group('research')]
sweep name cores="1":
    uv run snakemake --snakefile workflow/Snakefile --cores {{cores}} \
        results/complexity/{{name}}/complexity_per_dgp.parquet \
        results/complexity/{{name}}/complexity_by_setting.parquet

# Run the CV experiment through to aggregation
[group('research')]
cv cores="1":
    uv run snakemake --snakefile workflow/Snakefile --cores {{cores}} \
        results/cv/cv_basic/_cv_curve.parquet \
        results/cv/cv_basic/_cv_selection.parquet

# Run the PPC experiment through to its summary
[group('research')]
ppc cores="1":
    uv run snakemake --snakefile workflow/Snakefile --cores {{cores}} \
        results/ppc/ppc_result.json

# Render the DAG to SVG (requires graphviz's `dot`)
[group('research')]
dag:
    uv run snakemake --snakefile workflow/Snakefile --dag | dot -Tsvg > workflow_dag.svg

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
