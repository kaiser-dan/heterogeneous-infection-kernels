"""PPC rules: observed fit, then NREP independently parallelizable null fits
(wildcard r), then a summary.

`r` (null replicate index) is the ppc frame's only wildcard column;
every other column is identical across all rows, including for `ppc_observed`.
"""

from params_io import load_experiment
from model_select import model_tag_from_config_path

PARAMS_DIR = ROOT / "config" / "params"

PPC = load_experiment("ppc", PARAMS_DIR, filename_params="*")
PPC_FIXED = PPC.frame.iloc[0].to_dict()
NREP = int(PPC_FIXED["nrep"])

# np_config is fixed (non-wildcard) for this experiment
PPC_MODEL_TAG = model_tag_from_config_path(PPC_FIXED["np_config"], root=ROOT)

wildcard_constraints:
    r = r"\d+",


rule ppc_observed:
    output:
        f"{RESULTS}/ppc/{PPC_MODEL_TAG}/observed.npz"
    params:
        fixed=PPC_FIXED,
    script:
        "../scripts/run_ppc_observed.py"

rule ppc_null:
    input:
        f"{RESULTS}/ppc/{PPC_MODEL_TAG}/observed.npz"
    output:
        f"{RESULTS}/ppc/{PPC_MODEL_TAG}/nulls/{PPC.paramspace.wildcard_pattern}.npz"
    params:
        fixed=PPC_FIXED,
    script:
        "../scripts/run_ppc_null.py"

rule ppc_summary:
    input:
        observed=f"{RESULTS}/ppc/{PPC_MODEL_TAG}/observed.npz",
        nulls=[f"{RESULTS}/ppc/{PPC_MODEL_TAG}/nulls/{p}.npz" for p in PPC.paramspace.instance_patterns],
    output:
        result_json=f"{RESULTS}/ppc/{PPC_MODEL_TAG}/ppc_result.json",
        null_dist=f"{RESULTS}/ppc/{PPC_MODEL_TAG}/ppc_null_distribution.npz",
    params:
        fixed=PPC_FIXED,
    script:
        "../scripts/ppc_summary.py"
