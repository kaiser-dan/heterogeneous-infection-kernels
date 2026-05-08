
rule aggregate_sweep:
    """Aggregate all sweep results into a single complexity table."""
    input:
        INFERENCE_OUTPUTS
    output:
        f"results/{cfg_name}/complexity_table.parquet"
    params:
        result_dir = f"results/{cfg_name}"
    shell:
        "uv run python scripts/build_complexity_table.py --results-dir {params.result_dir} --output {output}"
