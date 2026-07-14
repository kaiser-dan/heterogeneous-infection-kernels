"""fit_model rule and 1d/2d measure_complexity rule, keyed on {sweep}."""

import json

from params_io import load_experiment

PARAMS_DIR = ROOT / "config" / "params"
SWEEP1D_NAMES = ["congress_omega_sweep"]
SWEEP2D_NAMES = ["beta_delta_sweep", "epsilon_omega_sweep"]
SWEEP_NAMES = SWEEP1D_NAMES + SWEEP2D_NAMES

wildcard_constraints:
    sweep = "|".join(SWEEP_NAMES),


SWEEPS = {
    name: load_experiment(name, PARAMS_DIR, filename_params="*", single_wildcard="params")
    for name in SWEEP_NAMES
}

# --- fit_model rule (shared across all sweeps) ---------------------------

def sweep_model_spec(wc):
    row = SWEEPS[wc.sweep].row(wc)
    return {
        "config": row["model_config"],
        "instance_params": json.loads(row["instance_params_json"]),
        "save_vars": json.loads(row["save_vars_json"]),
    }


def sweep_dgp(wc):
    row = SWEEPS[wc.sweep].row(wc)
    if wc.sweep in SWEEP1D_NAMES:
        return build_sweep_edgelist_dgp(row)
    else:
        return build_sweep_sbm_dgp(row)


rule fit_model:
    output:
        f"{RESULTS}/fits/{{sweep}}/{{params}}.npz"
    params:
        model_spec=sweep_model_spec,
        dgp=sweep_dgp,
    script:
        "../scripts/fit_model.py"

# --- measure_complexity_1d rule ------------------------------------------

def sweep_combos_1d(sweep_name):
    """Every (model, x, rep) combo for a sweep with the exact output path 
    each one maps to, for the `measure_complexity` rule.
    """
    ef = SWEEPS[sweep_name]
    axis_name = ef.wildcard_columns[2]
    combos = []
    for row, pattern in zip(
        ef.paramspace.dataframe.itertuples(index=False), ef.paramspace.instance_patterns
    ):
        row = row._asdict()
        combos.append({
            "file": f"{RESULTS}/fits/{sweep_name}/{pattern}.npz",
            "model": row["model"],
            "x": row[axis_name],
            "rep": row["rep"],
        })
    return combos


rule measure_complexity_1d:
    wildcard_constraints:
        sweep = "|".join(SWEEP1D_NAMES),
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
        combos=lambda wc: sweep_combos_1d(wc.sweep),
        axis_name=lambda wc: SWEEPS[wc.sweep].wildcard_columns[2],
    script:
        "../scripts/measure_complexity_1d.py"

# --- measure_complexity_2d rule ------------------------------------------

def sweep_combos_2d(sweep_name):
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


rule measure_complexity_2d:
    wildcard_constraints:
        sweep = "|".join(SWEEP2D_NAMES),
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
        combos=lambda wc: sweep_combos_2d(wc.sweep),
        axis1_name=lambda wc: SWEEPS[wc.sweep].wildcard_columns[2],
        axis2_name=lambda wc: SWEEPS[wc.sweep].wildcard_columns[3],
    script:
        "../scripts/measure_complexity_2d.py"
