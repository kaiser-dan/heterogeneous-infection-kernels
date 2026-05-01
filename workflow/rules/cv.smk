import re


def _cv_fold_dirs(wildcards):
    """Reconstruct per-fold inference output dirs for a given group instance ID.

    Returns posterior_samples.parquet paths (declared outputs of run_inference)
    so Snakemake can track the dependency correctly. The aggregate script
    receives the parent directories and looks for cv_held_out_loglik.parquet
    inside each one.
    """
    cv_folds = int(config.get("cv_folds", 0))
    group_id = wildcards.group_instance_id
    fold_files = []
    for fold in range(cv_folds):
        fold_part = f"cv_fold~{fold}"
        parts = group_id.split("__")
        parts_with_fold = sorted(parts + [fold_part])
        fold_instance_id = "__".join(parts_with_fold)
        fold_files.append(
            f"results/{wildcards.sweep_name}/{wildcards.data_source}/"
            f"{wildcards.data_name}/{fold_instance_id}/posterior_samples.parquet"
        )
    return fold_files


rule run_cv_aggregate:
    input:
        fold_files = _cv_fold_dirs,
        script     = "workflow/scripts/analysis/aggregate_cv.py",
    output:
        "results/{sweep_name}/{data_source}/{data_name}/{group_instance_id}/cv_summary.parquet"
    run:
        # Derive directories from the declared proxy input files
        fold_dirs = [str(Path(f).parent) for f in input.fold_files]
        fold_dir_args = " ".join(f'"{d}"' for d in fold_dirs)
        cmd = (
            f"uv run python {input.script} "
            f"--fold-dirs {fold_dir_args} "
            f"--output {output[0]}"
        )
        shell(cmd)
