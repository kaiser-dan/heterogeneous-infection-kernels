from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd()))

from workflow.lib.experiment_plan import build_workflow_context


WORKFLOW_CONTEXT = build_workflow_context(config=config, workflow=workflow)

cfg_name = WORKFLOW_CONTEXT.cfg_name
ACTIVE_CONFIGFILE = WORKFLOW_CONTEXT.active_configfile
SOURCE = WORKFLOW_CONTEXT.source
data_prep_data = WORKFLOW_CONTEXT.data_prep_data
DATA_CFG_NAMES = list(WORKFLOW_CONTEXT.data_cfg_names)
SYNTHETIC_INSTANCE_OVERRIDES = WORKFLOW_CONTEXT.synthetic_instance_overrides
TWITTER_PROCESSED_DIR = WORKFLOW_CONTEXT.twitter_processed_dir
SWEEP_ROWS = list(WORKFLOW_CONTEXT.sweep_rows)
INFERENCE_OUTPUTS = list(WORKFLOW_CONTEXT.inference_outputs)

# 2. Main Rules
rule all:
    """Default rule: Only prepare processed data."""
    input:
        TWITTER_PROCESSED_DIR,
        expand("data/processed/oo/sim_{sim_id}_backbone_disparity/exposures.parquet", sim_id=[165, 57, 118, 70, 76, 131, 66, 61, 72, 67]),

rule inference:
    """Explicitly run models defined in the config."""
    input:
        INFERENCE_OUTPUTS

# 3. Include Modular Rules
include: "workflow/rules/data_prep.smk"
include: "workflow/rules/inference.smk"
