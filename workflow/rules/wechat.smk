"""WeChat Moments rules: raw logs -> counts -> CV folds and full fits -> complexity and SBM communities."""

import json

from params_io import load_experiment

WECHAT_EXPERIMENT = "wechat"
WECHAT = load_experiment(WECHAT_EXPERIMENT, PARAMS_DIR, filename_params="*")
WECHAT_FIXED = WECHAT.frame.iloc[0].to_dict()
WECHAT_KS = sorted(WECHAT.frame["K"].unique())
WECHAT_DATA = "data/wechat"

WECHAT_FOLD_TEMPLATE = f"{RESULTS}/cv/{WECHAT_EXPERIMENT}/{WECHAT.paramspace.wildcard_pattern}_fold.parquet"
WECHAT_FOLD_FILES = [f"{RESULTS}/cv/{WECHAT_EXPERIMENT}/{p}_fold.parquet" for p in WECHAT.paramspace.instance_patterns]


rule wechat_download:
    output:
        f"{WECHAT_DATA}/raw/{{window}}.txt"
    wildcard_constraints:
        window = r"\d+",
    shell:
        "curl -sSfL -o {output} https://raw.githubusercontent.com/pkumobile/WMdata/master/{wildcards.window}.txt"


rule wechat_counts:
    input:
        expand(f"{WECHAT_DATA}/raw/{{window}}.txt", window=json.loads(WECHAT_FIXED["windows_json"]))
    output:
        f"{WECHAT_DATA}/processed/counts.npz",
        f"{WECHAT_DATA}/processed/wechat.edgelist",
    params:
        kmax=WECHAT_FIXED["kmax"],
    script:
        "../scripts/wechat_counts.py"


rule wechat_cv_task:
    input:
        f"{WECHAT_DATA}/processed/counts.npz"
    output:
        WECHAT_FOLD_TEMPLATE
    params:
        fixed=lambda wc: WECHAT.row(wc),
    threads: 4
    script:
        "../scripts/run_wechat_cv_task.py"


rule wechat_aggregate_cv:
    input:
        WECHAT_FOLD_FILES
    output:
        f"{RESULTS}/cv/{WECHAT_EXPERIMENT}/_cv_curve.parquet",
        f"{RESULTS}/cv/{WECHAT_EXPERIMENT}/_cv_selection.parquet",
    params:
        true_k_by_label={},
    script:
        "../scripts/aggregate_cv.py"


rule wechat_fit:
    input:
        f"{WECHAT_DATA}/processed/counts.npz"
    output:
        f"{RESULTS}/fits/{WECHAT_EXPERIMENT}/K~{{K}}.npz"
    params:
        fixed=lambda wc: {**WECHAT_FIXED, "K": int(wc.K)},
    threads: 4
    script:
        "../scripts/wechat_fit.py"


rule wechat_complexity:
    input:
        expand(f"{RESULTS}/fits/{WECHAT_EXPERIMENT}/K~{{K}}.npz", K=WECHAT_KS)
    output:
        f"{RESULTS}/complexity/{WECHAT_EXPERIMENT}/complexity.parquet"
    script:
        "../scripts/measure_complexity_wechat.py"


rule wechat_sbm:
    input:
        edgelist=f"{WECHAT_DATA}/processed/wechat.edgelist",
        fit=f"{RESULTS}/fits/{WECHAT_EXPERIMENT}/K~2.npz",
    output:
        f"{RESULTS}/communities/{WECHAT_EXPERIMENT}/sbm.npz"
    params:
        inits=WECHAT_FIXED["sbm_inits"],
        n_perm=WECHAT_FIXED["n_perm"],
        seed=WECHAT_FIXED["seed"],
    conda:
        "../envs/graph-tool.yaml"
    script:
        "../scripts/wechat_sbm.py"
