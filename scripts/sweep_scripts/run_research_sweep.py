import argparse
import itertools
import yaml
import subprocess
import os
import sys
import shlex
from pathlib import Path
import json

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def generate_combinations(sweep_config):
    keys = list(sweep_config.keys())
    values = list(sweep_config.values())
    combinations = list(itertools.product(*values))
    return keys, combinations

def main():
    parser = argparse.ArgumentParser(description="Run research sweep")
    parser.add_argument("config", help="Path to config file")
    parser.add_argument("--hpc", action="store_true", help="Submit to HPC via SLURM")
    parser.add_argument("--slurm-template", help="Path to SLURM template script")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    args = parser.parse_args()

    print(f"Loading config from {args.config}", flush=True)
    config = load_config(args.config)
    
    # Get sweep parameters, default to empty dict if not present
    sweep_params = config.get('sweep', {})
    print(f"Sweep params: {sweep_params}", flush=True)
    
    # Base paths
    # Assuming run from repo root
    repo_root = Path.cwd()
    
    # Determine input path based on config source
    if config.get('source') == 'oo':
        oo_conf = config.get('oo', {})
        sim_id = oo_conf.get('sim_id')
        method = oo_conf.get('backbone_method', 'disparity')
        # Standardized path
        exposure_summary = repo_root / "data" / "processed" / "oo" / f"sim_{sim_id}_backbone_{method}"
    else:
        # Twitter default
        intervention = config.get('twitter', {}).get('intervention', 'default')
        exposure_summary = repo_root / "data" / "processed" / "twitter" / intervention

    # Clean results dir - use name as base
    results_base = repo_root / "results" / config.get('name', 'sweep_results')
    results_base.mkdir(parents=True, exist_ok=True)

    if not sweep_params:
        # Single run
        keys = []
        combinations = [()]
    else:
        keys, combinations = generate_combinations(sweep_params)

    print(f"Generating {len(combinations)} combinations", flush=True)

    for i, combo in enumerate(combinations):
        param_dict = dict(zip(keys, combo))
        
        # Create a unique run name/ID
        desc_parts = []
        for k, v in param_dict.items():
            if k == 'stan_file':
                v_name = Path(v).stem.replace("model_", "").replace("non_parametric_", "")
                # Shorten common names for brevity
                v_name = v_name.replace("monotonic_empirical_bayes", "mono_eb")
                v_name = v_name.replace("simple_contagion", "simple")
                v_name = v_name.replace("complex_contagion", "complex")
                desc_parts.append(v_name)
            else:
                desc_parts.append(f"{k}{v}")
                
        run_id = "_".join(desc_parts) if desc_parts else "base_run"
        # Prepend index to ensure uniqueness if params overlap somehow
        run_id = f"{i:03d}_{run_id}"
        
        output_dir = results_base / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Serialize specific params for this run
        instance_params_json = json.dumps(param_dict)
        
        # Construct command
        cmd = [
            "uv", "run", "python", "src/models/inference.py",
            "--config", str(args.config),
            "--exposure-summary", str(exposure_summary),
            "--output-dir", str(output_dir),
            "--instance-params", instance_params_json
        ]
        
        # Properly quote arguments for shell execution
        cmd_str = " ".join(shlex.quote(str(x)) for x in cmd)
        
        if args.hpc:
            if not args.slurm_template:
                print("Error: --slurm-template required for --hpc")
                sys.exit(1)
            
            # Read template
            with open(args.slurm_template, 'r') as f:
                template = f.read()
            
            job_name = f"{config.get('name', 'sweep')}_{run_id}"
            slurm_script_path = output_dir / "submit.sh"
            
            # Replace placeholder if exists, otherwise append
            # But usually we just want to run the command.
            # Let's assume template handles SBATCH headers and activates env
            
            script_content = template
            if not script_content.endswith('\n'):
                script_content += '\n'
            
            script_content += f"\n# Command\n{cmd_str}\n"
            
            with open(slurm_script_path, 'w') as f:
                f.write(script_content)
            
            # Use sbatch
            submit_cmd = ["sbatch", "-J", job_name, "-o", str(output_dir / "slurm-%j.out"), str(slurm_script_path)]
            
            if args.dry_run:
                print(f"[Dry Run] Did not submit: {' '.join(submit_cmd)}", flush=True)
            else:
                print(f"Submitting job {job_name}...", flush=True)
                subprocess.run(submit_cmd, check=True)

        else:
            print(f"Running locally: {run_id}")
            if args.dry_run:
                print(f"[Dry Run] {cmd_str}")
            else:
                subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
