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
        exposures = "data/processed/oo/{data_name}.parquet",
        edges = lambda wildcards: f"data/processed/oo/{wildcards.data_name.replace('_exposures', '')}_edges.parquet"
        # Wait, data_name is "sim_165_backbone_disparity_exposures". 
        # The edges file is "sim_165_backbone_disparity_edges.parquet".
        # So if data_name has "_exposures", we need to replace it?
        # run_sweep.py generates data_name as "sim_..._backbone_..._exposures"
        # So we can use a lambda or string manip? Snakemake wildcards are strings.
    output:
        directory("data/summaries/oo/{data_name}.parquet") 
        # keeping .parquet extension for consistency with rule run_inference input pattern
    run:
        # Convert _exposures to _edges for the second input
        # Actually easier to fix inputs:
        import shutil
        from pathlib import Path
        
        # Parse sim/method from data_name if needed, or rely on parallel file naming
        # data_name = "sim_165_backbone_disparity_exposures"
        # derived edges = "sim_165_backbone_disparity_edges.parquet"
        
        base_name = wildcards.data_name.replace("_exposures", "")
        edges_src = Path(f"data/processed/oo/{base_name}_edges.parquet")
        
        # Check if exists (Snakemake usually handles this if we state it as input)
        # But since we can't easily express the input pattern transformation in 'input' block 
        # without a lambda or complex regex, we might assume they exist if the exposure exists?
        # Better: use proper inputs.
        
        out_dir = Path(output[0])
        out_dir.mkdir(parents=True, exist_ok=True)
        
        shutil.copy(input.exposures, out_dir / "infections.parquet")
        shutil.copy(edges_src, out_dir / "edges.parquet")
        
        # Propagate M and N
        m_src = Path(str(input.exposures).replace("_exposures.parquet", "_M.parquet"))
        n_src = Path(str(input.exposures).replace("_exposures.parquet", "_N.parquet"))
        if m_src.exists(): shutil.copy(m_src, out_dir / "M.parquet")
        if n_src.exists(): shutil.copy(n_src, out_dir / "N.parquet")

rule run_inference:
    input:
        summary = "data/summaries/{data_source}/{data_name}.parquet",
        config = "configs/{sweep_name}.yaml"
    output:
        samples = "results/{sweep_name}/{data_source}/{data_name}/{instance_id}/posterior_samples.parquet",
        summary = "results/{sweep_name}/{data_source}/{data_name}/{instance_id}/inference_summary.parquet"
    params:
        instance_params = lambda wildcards: {p.split('~')[0]: p.split('~')[1].replace('__', '/') for p in wildcards.instance_id.split('/')} if wildcards.instance_id != "base" else {}
    run:
        import json
        params_json = json.dumps(params.instance_params)
        shell(f"uv run python src/models/inference.py --config {input.config} --exposure-summary {input.summary} --output-dir results/{wildcards.sweep_name}/{wildcards.data_source}/{wildcards.data_name}/{wildcards.instance_id} --instance-params '{params_json}'")
