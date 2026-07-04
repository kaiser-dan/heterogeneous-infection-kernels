"""fit_model rule and measure_complexity rule, keyed on {sweep}."""

import json

from params_io import load_experiment

PARAMS_DIR = ROOT / "config" / "params"
SWEEP_NAMES = ["beta_delta_sweep", "epsilon_omega_sweep"]

wildcard_constraints:
    sweep = "|".join(SWEEP_NAMES),


SWEEPS = {
    name: load_experiment(name, PARAMS_DIR, filename_params="*", single_wildcard="params")
    for name in SWEEP_NAMES
}


def sweep_model_spec(wc):
    row = SWEEPS[wc.sweep].row(wc)
    return {
        "config": row["model_config"],
        "instance_params": json.loads(row["instance_params_json"]),
        "save_vars": json.loads(row["save_vars_json"]),
    }


def sweep_dgp(wc):
    return build_sweep_sbm_dgp(SWEEPS[wc.sweep].row(wc))


rule fit_model:
    output:
        f"{RESULTS}/fits/{{sweep}}/{{params}}.npz"
    params:
        model_spec=sweep_model_spec,
        dgp=sweep_dgp,
    script:
        "../scripts/fit_model.py"


def sweep_combos(sweep_name):
    """Every (model, x1, x2, rep) combo for a sweep with the exact output path 
    each one maps to, for the `measure_complexity` rule.
    """
    ef = SWEEPS[sweep_name]
    axis1_name, axis2_name = ef.wildcard_columns[2], ef.wildcard_columns[3]
    combos = []
    for row, pattern in zip(
        ef.paramspace.dataframe.itertuples(index=False), ef.paramspace.instance_patterns
    ):
        row = row._asdict()
        combos.append({
            "file": f"{RESULTS}/fits/{sweep_name}/{pattern}.npz",
            "model": row["model"],
            "x1": row[axis1_name],
            "x2": row[axis2_name],
            "rep": row["rep"],
        })
    return combos


rule measure_complexity:
    input:
        lambda wc: expand(
            f"{RESULTS}/fits/{{sweep}}/{{params}}.npz",
            sweep=wc.sweep,
            params=list(SWEEPS[wc.sweep].paramspace.instance_patterns),
        )
    output:
        f"{RESULTS}/complexity/{{sweep}}/complexity_per_dgp.parquet",
        f"{RESULTS}/complexity/{{sweep}}/complexity_by_setting.parquet",
    params:
        combos=lambda wc: sweep_combos(wc.sweep),
        axis1_name=lambda wc: SWEEPS[wc.sweep].wildcard_columns[2],
        axis2_name=lambda wc: SWEEPS[wc.sweep].wildcard_columns[3],
    script:
        "../scripts/measure_complexity.py"
