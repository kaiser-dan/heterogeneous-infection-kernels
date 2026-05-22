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
PPC_OUTPUTS = list(WORKFLOW_CONTEXT.ppc_outputs)
PPC_PAIR_MAP = WORKFLOW_CONTEXT.ppc_pair_map
PPC_CONFIG = WORKFLOW_CONTEXT.ppc_config
CV_AGGREGATE_OUTPUTS = list(WORKFLOW_CONTEXT.cv_aggregate_outputs)

# 2. Main Rules
def _all_inputs(_wildcards=None):
    """Return the data-prep targets appropriate for the configured source."""
    if SOURCE == "twitter":
        return [TWITTER_PROCESSED_DIR]
    if SOURCE == "oo":
        return expand(
            "data/processed/oo/sim_{sim_id}_backbone_disparity/exposures.parquet",
            sim_id=[165, 57, 118, 70, 76, 131, 66, 61, 72, 67],
        )
    if SOURCE == "synthetic":
        return expand(
            "data/processed/synthetic/{data_name}/exposures.parquet",
            data_name=DATA_CFG_NAMES,
        )
    return []


rule all:
    """Default rule: Only prepare processed data for the configured source."""
    input:
        _all_inputs,

rule inference:
    """Explicitly run models defined in the config."""
    input:
        INFERENCE_OUTPUTS

rule ppc:
    """Run configured posterior predictive checks."""
    input:
        PPC_OUTPUTS

rule cv_aggregate:
    """Aggregate k-fold CV scores per model config."""
    input:
        CV_AGGREGATE_OUTPUTS

# 3. Include Modular Rules
include: "workflow/rules/data_prep.smk"
include: "workflow/rules/inference.smk"
include: "workflow/rules/analysis.smk"
include: "workflow/rules/cv.smk"
