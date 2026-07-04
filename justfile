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

# Compile all Stan models in src/inference/
[group('setup')]
compile:
    uv run bin/compile_models.py

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
