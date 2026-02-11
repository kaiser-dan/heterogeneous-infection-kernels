wildcard_constraints:
    sweep_name="[^/]+",
    data_source="[^/]+",
    data_name="[^/]+",
    instance_id=".+"

rule generate_twitter_summary:
    input:
        "data/processed/twitter"
    output:
        "data/summaries/twitter/default_data_prep.parquet"
    run:
        import pandas as pd
        from pathlib import Path
        import os
        # Ensure directory exists
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        # Combine all A-matrices for Twitter
        p = Path(input[0])
        combined_rows = []
        for intervention_dir in p.glob("*"):
            if not intervention_dir.is_dir() or intervention_dir.name in ["full", "summaries"]:
                continue
            
            am_path = intervention_dir / "A_matrix.parquet"
            bv_path = intervention_dir / "B_vector.parquet"
            if not am_path.exists() or not bv_path.exists():
                continue
                
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
        summary_df.to_parquet(output[0])

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
        config = "configs/{sweep_name}.yaml"
    output:
        samples = "results/{sweep_name}/{data_source}/{data_name}/{instance_id}/posterior_samples.parquet",
        summary = "results/{sweep_name}/{data_source}/{data_name}/{instance_id}/inference_summary.parquet"
    run:
        import json
        
        instance_params = parse_instance_params(wildcards.instance_id)
        params_json = json.dumps(instance_params)
        # Escape curly braces for Snakemake's shell() function
        params_json_escaped = params_json.replace('{', '{{').replace('}', '}}')
        
        cmd = f"uv run python src/models/inference.py --config {input.config} --exposure-summary {input.summary} --output-dir results/{wildcards.sweep_name}/{wildcards.data_source}/{wildcards.data_name}/{wildcards.instance_id} --instance-params '{params_json_escaped}'"
        shell(cmd)
