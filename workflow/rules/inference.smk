wildcard_constraints:
    sweep_name="[^/]+",
    data_source="[^/]+",
    data_name="[^/]+",
    instance_id=".+",
    pair_id="[^/]+"

rule generate_twitter_summary:
    input:
        processed = lambda wildcards: TWITTER_PROCESSED_DIR,
        config = lambda wildcards: ACTIVE_CONFIGFILE
    output:
        directory("data/summaries/twitter/{data_name}.parquet")
    run:
        import shutil
        import pandas as pd
        from pathlib import Path

        p = Path(input.processed)
        out_dir = Path(output[0])
        out_dir.mkdir(parents=True, exist_ok=True)

        # Combine A/B matrices for Twitter
        combined_rows = []
        sources = []

        # Primary mode: per-intervention directories.
        for intervention_dir in p.glob("*"):
            if not intervention_dir.is_dir() or intervention_dir.name in ["full", "summaries"]:
                continue

            am_path = intervention_dir / "A_matrix.parquet"
            bv_path = intervention_dir / "B_vector.parquet"
            if am_path.exists() and bv_path.exists():
                sources.append((intervention_dir.name, am_path, bv_path))

        # Fallback mode: full-only preprocessing output.
        if not sources:
            full_dir = p / "full"
            am_path = full_dir / "A_matrix.parquet"
            bv_path = full_dir / "B_vector.parquet"
            if am_path.exists() and bv_path.exists():
                sources.append(("full", am_path, bv_path))

        for _, am_path, bv_path in sources:
            am = pd.read_parquet(am_path)
            bv = pd.read_parquet(bv_path)
            
            # Merge on user_id
            merged = am.merge(bv, on='user_id')
            
            # Exposure columns are everything except user_id and adopted
            exposure_cols = [c for c in merged.columns if c not in ['user_id', 'adopted']]
            
            # Convert to rows with A_array
            for row in merged.itertuples(index=False):
                # Construct A_array from exposure columns
                # row is a namedtuple, so we access by field name or index
                # it's safer to use the merging logic to get the values
                row_dict = dict(zip(merged.columns, row))
                A = [row_dict[col] for col in exposure_cols if row_dict[col] > 0]
                combined_rows.append({
                    'user_id': row_dict['user_id'],
                    'A_array': A,
                    'adopted': bool(row_dict['adopted'])
                })
        
        if not combined_rows:
            raise ValueError(f"No exposure data found in {p}")

        summary_df = pd.DataFrame(combined_rows)
        summary_df.to_parquet(out_dir / "summary.parquet", index=False)

        q = float(config.get("q", 1.0))
        metric = str(config.get("exposure_metric", "kappa"))
        full_dir = p / "full"

        def find_metric_matrix(kind: str) -> Path:
            exact = full_dir / f"{kind}_{metric}_q{q}.parquet"
            if exact.exists():
                return exact

            matches = []
            prefix = f"{kind}_{metric}_q"
            for candidate in full_dir.glob(f"{kind}_{metric}_q*.parquet"):
                stem = candidate.stem
                if not stem.startswith(prefix):
                    continue
                try:
                    found_q = float(stem[len(prefix):])
                except ValueError:
                    continue
                if abs(found_q - q) <= 1e-9:
                    matches.append(candidate)

            if not matches:
                raise FileNotFoundError(
                    f"Could not find {kind}_{metric} matrix for q={q} in {full_dir}"
                )
            return sorted(matches)[0]

        shutil.copy(find_metric_matrix("M"), out_dir / "M.parquet")
        shutil.copy(find_metric_matrix("N"), out_dir / "N.parquet")

rule generate_oo_summary:
    input:
        exposures = "data/processed/oo/{data_name}/exposures.parquet",
        edges = "data/processed/oo/{data_name}/edges.parquet",
        M = "data/processed/oo/{data_name}/M.parquet",
        N = "data/processed/oo/{data_name}/N.parquet"
    output:
        directory("data/summaries/oo/{data_name}.parquet") 
    run:
        import shutil
        from pathlib import Path
        
        out_dir = Path(output[0])
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy files to expected names for inference
        # inference.py expects 'infections.parquet' (legacy name?) or we can stick to exposures
        # generate_stan_matrices_oo expects 'infections.parquet'
        shutil.copy(input.exposures, out_dir / "infections.parquet")
        shutil.copy(input.edges, out_dir / "edges.parquet")
        
        if Path(input.M).exists(): shutil.copy(input.M, out_dir / "M.parquet")
        if Path(input.N).exists(): shutil.copy(input.N, out_dir / "N.parquet")


rule generate_synthetic_processed:
    input:
        script = "workflow/scripts/simulate/generate_synthetic_data.py"
    params:
        config = lambda wildcards: ACTIVE_CONFIGFILE
    output:
        M = "data/processed/synthetic/{data_name}/M.parquet",
        N = "data/processed/synthetic/{data_name}/N.parquet",
        edges = "data/processed/synthetic/{data_name}/edges.parquet",
        exposures = "data/processed/synthetic/{data_name}/exposures.parquet",
        kernels = "data/processed/synthetic/{data_name}/kernels.parquet",
        metadata = "data/processed/synthetic/{data_name}/metadata.parquet"
    run:
        import json
        import shlex

        overrides = SYNTHETIC_INSTANCE_OVERRIDES.get(wildcards.data_name, {})
        overrides_json = json.dumps(overrides, separators=(",", ":"))
        # Snakemake's shell() uses format-style expansion; escape braces in JSON payloads.
        overrides_json_escaped = overrides_json.replace("{", "{{").replace("}", "}}")
        cmd = (
            f"uv run python {input.script} "
            f"--config {params.config} "
            f"--output-dir data/processed/synthetic/{wildcards.data_name} "
            f"--data-name {wildcards.data_name} "
            f"--synthetic-overrides {shlex.quote(overrides_json_escaped)}"
        )
        shell(cmd)


rule generate_synthetic_summary:
    input:
        M = "data/processed/synthetic/{data_name}/M.parquet",
        N = "data/processed/synthetic/{data_name}/N.parquet",
        edges = "data/processed/synthetic/{data_name}/edges.parquet",
        exposures = "data/processed/synthetic/{data_name}/exposures.parquet",
        metadata = "data/processed/synthetic/{data_name}/metadata.parquet",
    output:
        directory("data/summaries/synthetic/{data_name}.parquet")
    run:
        import shutil
        from pathlib import Path

        out_dir = Path(output[0])
        out_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(input.edges, out_dir / "edges.parquet")
        shutil.copy(input.exposures, out_dir / "infections.parquet")
        shutil.copy(input.M, out_dir / "M.parquet")
        shutil.copy(input.N, out_dir / "N.parquet")
        shutil.copy(input.metadata, out_dir / "metadata.parquet")

def parse_instance_params(instance_id):
    if instance_id == "base":
        return dict()
    params_dict = dict()
    for p in instance_id.split('/'):
        key_val = p.split('~')
        value = key_val[1].replace('__', '/')
        # Try to convert to int or float if possible
        try:
            if '.' in value:
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            # Keep as string if not a number
            pass
        params_dict[key_val[0]] = value
    return params_dict

rule run_inference:
    input:
        summary = "data/summaries/{data_source}/{data_name}.parquet",
        config = lambda wildcards: ACTIVE_CONFIGFILE
    output:
        samples = "results/{sweep_name}/{data_source}/{data_name}/{instance_id}/posterior_samples.parquet",
        summary = "results/{sweep_name}/{data_source}/{data_name}/{instance_id}/inference_summary.parquet"
    threads: 4
    run:
        import json
        
        instance_params = parse_instance_params(wildcards.instance_id)
        params_json = json.dumps(instance_params)
        # Escape curly braces for Snakemake's shell() function
        params_json_escaped = params_json.replace('{', '{{').replace('}', '}}')
        
        cmd = f"uv run python src/models/inference.py --config {input.config} --exposure-summary {input.summary} --output-dir results/{wildcards.sweep_name}/{wildcards.data_source}/{wildcards.data_name}/{wildcards.instance_id} --instance-params '{params_json_escaped}'"
        shell(cmd)


rule run_ppc_non_nested:
    input:
        script = "workflow/scripts/analysis/run_ppc_non_nested.py",
        summary = "data/summaries/{data_source}/{data_name}.parquet",
        simple_samples = lambda wildcards: (
            f"results/{wildcards.sweep_name}/{wildcards.data_source}/{wildcards.data_name}/"
            f"{PPC_PAIR_MAP[wildcards.pair_id]['simple_instance_id']}/posterior_samples.parquet"
        ),
    output:
        details = "results/{sweep_name}/{data_source}/{data_name}/ppc/{pair_id}/non_nested_details.parquet",
        summary = "results/{sweep_name}/{data_source}/{data_name}/ppc/{pair_id}/non_nested_summary.parquet"
    run:
        import json
        import shlex

        stats = PPC_CONFIG.get("non_nested_stats", PPC_CONFIG.get("stats", ["deviance", "neg_loglik", "l1_rate"]))
        if not isinstance(stats, list):
            raise ValueError("analysis.ppc.stats must be a list.")
        stats_arg = ",".join(str(s) for s in stats)
        draws = int(PPC_CONFIG.get("draws", 100))
        seed = int(PPC_CONFIG.get("seed", 123))

        cmd = (
            f"uv run python {input.script} "
            f"--summary-dir {input.summary} "
            f"--simple-posterior {input.simple_samples} "
            f"--details-output {output.details} "
            f"--summary-output {output.summary} "
            f"--draws {draws} "
            f"--seed {seed} "
            f"--stats {shlex.quote(stats_arg)}"
        )
        shell(cmd)


rule run_ppc_nested_refit:
    input:
        script = "workflow/scripts/analysis/run_ppc_nested_refit.py",
        summary = "data/summaries/{data_source}/{data_name}.parquet",
        config = lambda wildcards: ACTIVE_CONFIGFILE,
        simple_obs_samples = lambda wildcards: (
            f"results/{wildcards.sweep_name}/{wildcards.data_source}/{wildcards.data_name}/"
            f"{PPC_PAIR_MAP[wildcards.pair_id]['simple_instance_id']}/posterior_samples.parquet"
        ),
        nonparam_obs_samples = lambda wildcards: (
            f"results/{wildcards.sweep_name}/{wildcards.data_source}/{wildcards.data_name}/"
            f"{PPC_PAIR_MAP[wildcards.pair_id]['nonparam_instance_id']}/posterior_samples.parquet"
        ),
    output:
        details = "results/{sweep_name}/{data_source}/{data_name}/ppc/{pair_id}/nested_details.parquet",
        summary = "results/{sweep_name}/{data_source}/{data_name}/ppc/{pair_id}/nested_summary.parquet"
    run:
        import json
        import shlex

        pair = PPC_PAIR_MAP[wildcards.pair_id]
        simple_params = parse_instance_params(pair["simple_instance_id"])
        nonparam_params = parse_instance_params(pair["nonparam_instance_id"])
        simple_params_json = json.dumps(simple_params, separators=(",", ":"))
        nonparam_params_json = json.dumps(nonparam_params, separators=(",", ":"))
        simple_params_json_escaped = simple_params_json.replace("{", "{{").replace("}", "}}")
        nonparam_params_json_escaped = nonparam_params_json.replace("{", "{{").replace("}", "}}")

        draws = int(PPC_CONFIG.get("nested_draws", PPC_CONFIG.get("draws", 20)))
        seed = int(PPC_CONFIG.get("seed", 123))
        nested_stats = PPC_CONFIG.get("nested_stats", PPC_CONFIG.get("nested_statistic", "mahal_non_centrality"))
        if isinstance(nested_stats, list):
            nested_stats_arg = ",".join(str(s) for s in nested_stats)
        elif isinstance(nested_stats, str):
            nested_stats_arg = nested_stats
        else:
            raise ValueError("analysis.ppc.nested_stats must be a list or string.")

        cmd = (
            f"uv run python {input.script} "
            f"--summary-dir {input.summary} "
            f"--config {input.config} "
            f"--simple-observed-posterior {input.simple_obs_samples} "
            f"--nonparam-observed-posterior {input.nonparam_obs_samples} "
            f"--simple-instance-params {shlex.quote(simple_params_json_escaped)} "
            f"--nonparam-instance-params {shlex.quote(nonparam_params_json_escaped)} "
            f"--details-output {output.details} "
            f"--summary-output {output.summary} "
            f"--draws {draws} "
            f"--seed {seed} "
            f"--stats {shlex.quote(nested_stats_arg)}"
        )
        shell(cmd)
