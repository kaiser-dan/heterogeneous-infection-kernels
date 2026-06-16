from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd()))

from workflow.lib.experiment_plan import build_workflow_context

WORKFLOW_CONTEXT = build_workflow_context(config=config, workflow=workflow)

cfg_name                     = WORKFLOW_CONTEXT.cfg_name
ACTIVE_CONFIGFILE            = WORKFLOW_CONTEXT.active_configfile
DATA_CFG_NAMES               = list(WORKFLOW_CONTEXT.data_cfg_names)
SYNTHETIC_INSTANCE_OVERRIDES = WORKFLOW_CONTEXT.synthetic_instance_overrides
SWEEP_ROWS                   = list(WORKFLOW_CONTEXT.sweep_rows)
INFERENCE_OUTPUTS            = list(WORKFLOW_CONTEXT.inference_outputs)
CV_AGGREGATE_OUTPUTS         = list(WORKFLOW_CONTEXT.cv_aggregate_outputs)
COMPLEXITY_OUTPUTS           = list(WORKFLOW_CONTEXT.complexity_outputs)


rule all:
    """Prepare all synthetic datasets."""
    input:
        expand(
            "data/synthetic/{data_name}/M.npy",
            data_name=DATA_CFG_NAMES,
        )

rule inference:
    """Run all configured model fits."""
    input:
        INFERENCE_OUTPUTS

rule cv_aggregate:
    """Aggregate k-fold CV scores per model config."""
    input:
        CV_AGGREGATE_OUTPUTS

rule complexity:
    """Compute complexity scores for all fitted model pairs."""
    input:
        COMPLEXITY_OUTPUTS


include: "workflow/rules/simulation.smk"
include: "workflow/rules/inference.smk"
include: "workflow/rules/cv.smk"
include: "workflow/rules/analysis.smk"