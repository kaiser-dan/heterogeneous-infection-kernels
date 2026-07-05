"""CV rules. Paramspace-driven fan-out over (dgp_label, K, rep, fold)."""

import json

from params_io import load_experiment
from model_select import model_tag_from_config_path

PARAMS_DIR = ROOT / "config" / "params"
CV_EXPERIMENT = "cv_basic"

CV = load_experiment(CV_EXPERIMENT, PARAMS_DIR, filename_params="*")
DGP_LABELS = sorted(CV.frame["dgp_label"].unique())

# model_config is fixed (non-wildcard) for this experiment
CV_MODEL_TAG = model_tag_from_config_path(CV.frame["model_config"].iloc[0], root=ROOT)

wildcard_constraints:
    dgp_label = "|".join(DGP_LABELS),
    K = r"\d+",
    rep = r"\d+",
    fold = r"\d+",


CV_FOLD_TEMPLATE = f"{RESULTS}/cv/{CV_EXPERIMENT}/{CV_MODEL_TAG}/{CV.paramspace.wildcard_pattern}_fold.parquet"
CV_FOLD_FILES = [f"{RESULTS}/cv/{CV_EXPERIMENT}/{CV_MODEL_TAG}/{p}_fold.parquet" for p in CV.paramspace.instance_patterns]


rule run_cv_task:
    input:
        kernel_files=lambda wc: cv_kernel_files(json.loads(CV.row(wc)["kernels_json"]))
    output:
        CV_FOLD_TEMPLATE
    params:
        dgp=lambda wc: build_cv_dgp(CV.row(wc), root=ROOT),
        fixed=lambda wc: CV.row(wc),
    script:
        "../scripts/run_cv_task.py"


rule aggregate_cv:
    input:
        CV_FOLD_FILES
    output:
        f"{RESULTS}/cv/{CV_EXPERIMENT}/{CV_MODEL_TAG}/_cv_curve.parquet",
        f"{RESULTS}/cv/{CV_EXPERIMENT}/{CV_MODEL_TAG}/_cv_selection.parquet",
    params:
        true_k_by_label=CV.frame.drop_duplicates("dgp_label").set_index("dgp_label")["true_K"].to_dict(),
    script:
        "../scripts/aggregate_cv.py"
