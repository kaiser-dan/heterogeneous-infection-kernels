import yaml
import pandas as pd
from itertools import product
from pathlib import Path

# 1. Determine Workflow Mode and Build Experiment Grid
# Snakemake automatically populates the 'config' dict from --configfile and --config
cfg_name = config.get('name', 'default_run')

# Resolve Data Source and Config
# We still might want to load the data_config to get the SOURCE
data_config_path = config.get('data_config', 'configs/default_data_prep.yaml')
with open(data_config_path, 'r') as f:
    data_prep_data = yaml.safe_load(f)

SOURCE = config.get('source', data_prep_data.get('source', 'twitter'))

if SOURCE == 'oo':
    sim_id = config.get('oo', {}).get('sim_id', 165)
    method = config.get('oo', {}).get('backbone_method', 'disparity')
    DATA_CFG_NAME = f"sim_{sim_id}_backbone_{method}_exposures"
else:
    DATA_CFG_NAME = Path(data_config_path).stem

INFERENCE_OUTPUTS = []

def get_instance_dir(row_dict):
    parts = []
    # Sort keys for deterministic paths
    for k in sorted(row_dict.keys()):
        v = str(row_dict[k])
        # Strip path and extension for stan_file to keep dir clean
        if k == 'stan_file':
            v = Path(v).stem
        parts.append(f"{k}~{v}")
    return "/".join(parts)

if "sweep" in config:
    keys = config["sweep"].keys()
    values = config["sweep"].values()
    grid = [dict(zip(keys, v)) for v in product(*values)]
    
    for row in grid:
        instance_path = get_instance_dir(row)
        INFERENCE_OUTPUTS.append(f"results/{cfg_name}/{SOURCE}/{DATA_CFG_NAME}/{instance_path}/posterior_samples.parquet")
        INFERENCE_OUTPUTS.append(f"results/{cfg_name}/{SOURCE}/{DATA_CFG_NAME}/{instance_path}/inference_summary.parquet")
else:
    INFERENCE_OUTPUTS.append(f"results/{cfg_name}/{SOURCE}/{DATA_CFG_NAME}/base/posterior_samples.parquet")
    INFERENCE_OUTPUTS.append(f"results/{cfg_name}/{SOURCE}/{DATA_CFG_NAME}/base/inference_summary.parquet")

# 2. Main Rules
rule all:
    """Default rule: Only prepare processed data."""
    input:
        "data/processed/twitter",
        expand("data/processed/oo/sim_{sim_id}_backbone_disparity_exposures.parquet", sim_id=[165, 57, 118, 70, 76, 131, 66, 61, 72, 67]),

rule inference:
    """Explicitly run models defined in the config."""
    input:
        INFERENCE_OUTPUTS

# 3. Include Modular Rules
include: "workflow/rules/data_prep.smk"
include: "workflow/rules/inference.smk"
