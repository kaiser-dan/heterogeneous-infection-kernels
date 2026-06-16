# workflow/rules/analysis.smk

wildcard_constraints:
    group_instance_id = "[^/]+",


def _complexity_input_posteriors(wildcards):
    """Resolve posterior paths for the reference and target models.

    group_instance_id encodes everything except stan_file (and cv_fold), e.g.
    K~2. We inject each stan_file to get the two posterior paths. Which model
    plays each role is controlled by complexity.ref_model and
    complexity.target_model in config.yaml.
    """
    ref_model    = config.get("complexity", {}).get("ref_model",    "mixture_simple_contagion")
    target_model = config.get("complexity", {}).get("target_model", "non_parametric_monotonic_empirical_bayes")
    group_id     = wildcards.group_instance_id.replace("__", "/")
    base         = f"results/{wildcards.sweep_name}/{wildcards.data_name}"
    ref_parts    = sorted(group_id.split("/") + [f"stan_file~{ref_model}"])
    target_parts = sorted(group_id.split("/") + [f"stan_file~{target_model}"])
    return {
        "ref":    f"{base}/{'/'.join(ref_parts)}/posterior_samples.parquet",
        "target": f"{base}/{'/'.join(target_parts)}/posterior_samples.parquet",
    }


def _complexity_kmax(wildcards):
    """Read kmax from the dataset metadata at rule evaluation time."""
    import pandas as pd
    meta = pd.read_parquet(f"data/synthetic/{wildcards.data_name}/metadata.parquet")
    return int(meta["kmax"].iloc[0])


def _complexity_K(wildcards):
    """Extract K from the group_instance_id wildcard."""
    for part in wildcards.group_instance_id.replace("__", "/").split("/"):
        if part.startswith("K~"):
            return int(part.split("~")[1])
    return 1


rule run_complexity:
    """Compute Mahalanobis complexity scores for a reference/target model pair."""
    input:
        unpack(_complexity_input_posteriors),
        script = "workflow/scripts/run_complexity.py",
    output:
        "results/{sweep_name}/{data_name}/{group_instance_id}/complexity_scores.parquet",
    run:
        K    = _complexity_K(wildcards)
        kmax = _complexity_kmax(wildcards)
        out_dir = str(Path(output[0]).parent)
        shell(
            f"uv run python {input.script} "
            f"--posterior-ref {input.ref} "
            f"--posterior-target {input.target} "
            f"--output-dir {out_dir} "
            f"--K {K} "
            f"--kmax {kmax}"
        )


rule aggregate_sweep:
    """Collect all complexity_scores.parquet files into one sweep-level table."""
    input:
        COMPLEXITY_OUTPUTS,
    output:
        f"results/{cfg_name}/complexity_table.parquet",
    run:
        import pandas as pd
        from pathlib import Path

        dfs = []
        for path in input:
            df = pd.read_parquet(path)
            parts = Path(path).parts  # results/sweep/data_name/group_id/file
            df["data_name"]         = parts[2]
            df["group_instance_id"] = parts[3]
            dfs.append(df)

        pd.concat(dfs, ignore_index=True).to_parquet(output[0], index=False)
        print(f"Wrote complexity table to {output[0]}")
