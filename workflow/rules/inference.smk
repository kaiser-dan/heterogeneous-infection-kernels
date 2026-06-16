# workflow/rules/inference.smk

wildcard_constraints:
    sweep_name   = "[^/]+",
    data_name    = "[^/]+",
    instance_id  = ".+",


def parse_instance_params(instance_id: str) -> dict:
    """Decode an instance_id into a dict."""
    if instance_id == "base":
        return {}
    params = {}
    for part in instance_id.split("/"):
        key, val = part.split("~", 1)
        val = val.replace("__", "/")
        try:
            val = int(val) if "." not in val else float(val)
        except ValueError:
            pass
        params[key] = val
    return params


rule run_synthetic_inference:
    """Fit a Stan model to one synthetic dataset with one parameter configuration."""
    input:
        M      = "data/synthetic/{data_name}/M.npy",
        N      = "data/synthetic/{data_name}/N.npy",
        config = lambda wildcards: ACTIVE_CONFIGFILE,
    output:
        samples = "results/{sweep_name}/{data_name}/{instance_id}/posterior_samples.parquet",
        summary = "results/{sweep_name}/{data_name}/{instance_id}/inference_summary.parquet",
    threads: 4
    run:
        import json
        import shlex

        params      = parse_instance_params(wildcards.instance_id)
        params_json = shlex.quote(json.dumps(params, separators=(",", ":")))
        shell(
            f"uv run python -m src.inference.inference "
            f"--config {input.config} "
            f"--M {input.M} "
            f"--N {input.N} "
            f"--output-dir results/{wildcards.sweep_name}/{wildcards.data_name}/{wildcards.instance_id} "
            f"--instance-params {params_json}"
        )
