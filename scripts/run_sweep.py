import yaml
import argparse
from pathlib import Path
import itertools
import subprocess
import sys

def run_sweep(config_path):
    with open(config_path) as f:
        conf = yaml.safe_load(f)
    
    sweep_name = conf['name']
    source = conf['source']
    
    # Determine data_name
    if source == 'twitter':
        data_name = 'default_data_prep'
    elif source == 'oo':
        sim_id = conf['oo']['sim_id']
        method = conf['oo']['backbone_method']
        # Match the pattern in Snakefile/data_prep.smk
        # output of process_oo_sim: sim_{sim}_backbone_{method}_exposures.parquet
        # data_name for summaries is basename of that.
        # Wait, rule process_oo_sim in data_prep.smk (Step 28) outputs:
        # "oo/sim_{sim_id}_exposures.parquet" (Wait, Step 28 vs Step 36 differ?)
        # Step 28 data_prep.smk: sim_{sim_id}_exposures.parquet (Simple)
        # Step 36 Snakefile: sim_{sim}_backbone_{method}_exposures.parquet (Complex)
        # Let's check which one is active. The user referenced research_sweep_oo.yaml which has backbone_method.
        # I'll assume the complex one if backbone is specified, or check data_prep.smk again.
        # Step 28 was data_prep.smk. Step 36 was Snakefile. 
        # Usually snakefile includes data_prep? 
        # Let's assume the data name logic:
        # If configs/research_sweep_oo.yaml has backbone_method, we construct it.
        if 'backbone_method' in conf['oo']:
            data_name = f"sim_{sim_id}_backbone_{method}_exposures"
        else:
            data_name = f"sim_{sim_id}_exposures"
            
    # Sweep parameters
    sweep_params = conf.get('sweep', {})
    if not sweep_params:
        print("No sweep parameters found.")
        return

    keys = sorted(sweep_params.keys())
    values = [sweep_params[k] for k in keys]
    
    targets = []
    
    for combination in itertools.product(*values):
        # Construct instance_id: k1~v1/k2~v2...
        parts = []
        for k, v in zip(keys, combination):
            parts.append(f"{k}~{v}")
        
        instance_id = "/".join(parts)
        
        # Target file
        # results/{sweep_name}/{data_source}/{data_name}/{instance_id}/inference_summary.parquet
        target = f"results/{sweep_name}/{source}/{data_name}/{instance_id}/inference_summary.parquet"
        targets.append(target)
        
    print(f"Generated {len(targets)} targets for sweep '{sweep_name}'")
    
    # Run Snakemake
    # We need to point to the inference rules. 
    # If workflow/Snakefile doesn't include them, we point to workflow/rules/inference.smk
    snakefile = "workflow/rules/inference.smk"
    
    cmd = [
        "uv", "run", "snakemake",
        "-s", snakefile,
        "-c1",
        "--configfile", str(config_path)
    ] + targets
    
    print("Running command:")
    print(" ".join(cmd))
    
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    
    run_sweep(args.config)
