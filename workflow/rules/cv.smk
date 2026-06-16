# workflow/rules/cv.smk

wildcard_constraints:
    group_instance_id = "[^/]+",


def _cv_fold_dirs(wildcards):
    """Reconstruct per-fold posterior_samples.parquet paths for a group instance.

    group_instance_id is the instance_id with cv_fold stripped and slashes
    encoded as __, e.g. K~2__stan_file~mixture_simple_contagion. This function
    decodes it, injects each cv_fold~N, and returns the expected output paths
    so Snakemake can track the dependency on run_inference.
    """
    cv_folds = int(config.get("cv_folds", 0))
    group_id = wildcards.group_instance_id.replace("__", "/")
    fold_files = []
    for fold in range(cv_folds):
        parts = sorted(group_id.split("/") + [f"cv_fold~{fold}"])
        fold_instance_id = "/".join(parts)
        fold_files.append(
            f"results/{wildcards.sweep_name}/{wildcards.data_name}"
            f"/{fold_instance_id}/posterior_samples.parquet"
        )
    return fold_files


rule run_cv_aggregate:
    """Aggregate held-out log-likelihoods across CV folds into a single summary."""
    input:
        fold_files = _cv_fold_dirs,
        script     = "workflow/scripts/aggregate_cv.py",
    output:
        "results/{sweep_name}/{data_name}/{group_instance_id}/cv_summary.parquet",
    run:
        from pathlib import Path
        fold_dirs    = " ".join(f'"{Path(f).parent}"' for f in input.fold_files)
        shell(
            f"uv run python {input.script} "
            f"--fold-dirs {fold_dirs} "
            f"--output {output[0]} "
            f'--group-id "{wildcards.group_instance_id}"'
        )
