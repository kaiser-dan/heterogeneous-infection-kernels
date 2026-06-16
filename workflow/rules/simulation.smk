# workflow/rules/simulation.smk
# Generates synthetic datasets. One rule per dataset name, driven by the
# sim_sweep grid in config. Outputs live in data/synthetic/{data_name}/

rule generate_synthetic:
    """Simulate a synthetic SBM+SIS dataset and write M/N matrices."""
    input:
        script = "workflow/scripts/generate_synthetic_data.py",
        config = lambda wildcards: ACTIVE_CONFIGFILE,
    output:
        M         = "data/synthetic/{data_name}/M.npy",
        N         = "data/synthetic/{data_name}/N.npy",
        edges     = "data/synthetic/{data_name}/edges.parquet",
        exposures = "data/synthetic/{data_name}/exposures.parquet",
        kernels   = "data/synthetic/{data_name}/kernels.parquet",
        metadata  = "data/synthetic/{data_name}/metadata.parquet",
    run:
        import json
        import shlex

        overrides = SYNTHETIC_INSTANCE_OVERRIDES.get(wildcards.data_name, {})
        overrides_json = shlex.quote(json.dumps(overrides, separators=(",", ":")))
        shell(
            f"uv run python {input.script} "
            f"--config {input.config} "
            f"--output-dir data/synthetic/{wildcards.data_name} "
            f"--data-name {wildcards.data_name} "
            f"--synthetic-overrides {overrides_json}"
        )
