import yaml
import argparse
from pathlib import Path
import itertools
import subprocess
import sys

def run_sweep(config_path, profile=None):
    with open(config_path) as f:
        conf = yaml.safe_load(f)
    
    sweep_name = conf['name']
    source = conf['source']
    
    # Determine data_name
    if source == 'twitter':
        data_name = 'default_data_prep'
    elif source == 'oo':
        sim_id = conf['oo']['sim_id']
        method = conf['oo'].get('backbone_method', 'raw')
        data_name = f"sim_{sim_id}_backbone_{method}_exposures"
            
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
            # Strip path and extension for stan_file to keep dir clean
            if k == 'stan_file':
                v = Path(v).stem
            v_safe = str(v).replace('/', '__')
            parts.append(f"{k}~{v_safe}")
        
        instance_id = "/".join(parts)
        
        # Target file
        # results/{sweep_name}/{data_source}/{data_name}/{instance_id}/inference_summary.parquet
        target = f"results/{sweep_name}/{source}/{data_name}/{instance_id}/inference_summary.parquet"
        targets.append(target)
        
    print(f"Generated {len(targets)} targets for sweep '{sweep_name}'")
    
    # Run Snakemake
    # Use the root Master Snakefile for all entry points
    snakefile = "Snakefile"
    
    cmd = [
        "uv", "run", "snakemake",
    ] + targets + [
        "-s", snakefile,
        "--configfile", str(config_path)
    ]

    if profile:
        cmd.extend(["--profile", profile])
    else:
        # Default fallback if no profile is given, though we likely usually want one
        # Or maybe -c1 if no profile? Let's leave -c1 separate or assume profile handles it.
        # Original code had -c1 hardcoded. Let's keep -c1 only if no profile is specified.
        cmd.append("-c1")
    
    print("Running command:")
    print(" ".join(cmd))
    
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--profile", type=str, help="Snakemake profile to use (e.g. slurm, local)")
    args = parser.parse_args()
    
    run_sweep(args.config, args.profile)
