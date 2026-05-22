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

CV_AGGREGATE_OUTPUTS = []
_cv_folds = int(config.get("cv_folds", 0))
if _cv_folds > 0 and "sweep" in config:
    _seen_cv = set()
    for _cv_row in build_grid(config["sweep"]):
        _group_row = {k: v for k, v in _cv_row.items() if k != "cv_fold"}
        _group_instance_id = encode_instance_id(get_instance_dir(_group_row))
        for data_name in DATA_CFG_NAMES:
            _cv_path = f"results/{cfg_name}/{SOURCE}/{data_name}/{_group_instance_id}/cv_summary.parquet"
            if _cv_path not in _seen_cv:
                _seen_cv.add(_cv_path)
                CV_AGGREGATE_OUTPUTS.append(_cv_path)

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
