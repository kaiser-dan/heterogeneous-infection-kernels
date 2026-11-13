"""CV sweep: CV model selection across a grid of cells, one twoSimple dgp and one
oneMatched dgp per (cell, rep). See config/cv_sweep_frame.py.

  stage 1 (cv_sweep_stage1_fit -> matched_kernel) - K=1 np fit to the twoSimple data of a
    (cell, rep), then its posterior-mean kernel. Always fitted here, under results/cv/<cv_sweep>/stage1/;
    nothing is read from results/fits/ (rule fit_model), even where a sweep has the same DGP.
  stage 2 (run_cv_sweep_task) - one CV fold for one (cell, dgp, K, rep).
  stage 3 (aggregate_cv_sweep) - per-rep CV curves and 1-SE selections, per-cell summary.

Uses RESULTS, ROOT and build_sweep_sbm_dgp from the Snakefile; independent of the other rule files.
"""

import json

from params_io import load_experiment

CV_SWEEP_NAMES = ["cv_epsilon_omega_sweep", "cv_beta_sweep"]
CV_SWEEP_PARAMS_DIR = ROOT / "config" / "params"
CV_SWEEP_N_NONAXIS_WILDCARDS = 4  # dgp, K, rep, fold follow the cell axes

CVS = {
    name: load_experiment(name, CV_SWEEP_PARAMS_DIR, filename_params="*", single_wildcard="params")
    for name in CV_SWEEP_NAMES
}

# build side look-up since stage 1 jobs keyed by (cell, rep) only
def _cell_rows(name):
    """(cell, rep) -> one representative frame row (the DGP is the same for every dgp/K/fold)."""
    first = CVS[name].frame.drop_duplicates(["cell", "rep"])
    return {(r["cell"], int(r["rep"])): r for r in first.to_dict("records")}

CV_SWEEP_CELL_ROWS = {name: _cell_rows(name) for name in CV_SWEEP_NAMES}

# gives matched kernel's fpath
def cv_sweep_kernel_path(name, cell, rep):
    return f"{RESULTS}/cv/{name}/kernels/{cell}_rep~{int(rep)}.txt"

# returns swept axes (e.g., ["epsilon", "omega"])
def cv_sweep_axes(name):
    return CVS[name].wildcard_columns[:-CV_SWEEP_N_NONAXIS_WILDCARDS]

# turns a job's wildcards into full table row
def cv_sweep_row(wc):
    return CVS[wc.cv_sweep].row(wc)


def cv_sweep_dgp(wc):
    row = cv_sweep_row(wc)
    dgp = build_sweep_sbm_dgp(row)  # default: twoSimple DGP (same graph/state/seeds as the stage-1 fit)
    if row["dgp"] == "oneMatched":
        # swap in single np kernel read from matched-kernel file
        kernel = cv_sweep_kernel_path(wc.cv_sweep, row["cell"], row["rep"])
        dgp["kernels"] = [{"type": "nonparametric", "values_file": str(ROOT / kernel)}]
        dgp["assignment"] = {}
    return dgp


def cv_sweep_kernel_input(wc):
    """A oneMatched fold job needs its matched kernel; a twoSimple one needs nothing."""
    row = cv_sweep_row(wc)
    if row["dgp"] == "oneMatched":
        return [cv_sweep_kernel_path(wc.cv_sweep, row["cell"], row["rep"])]
    return []

# lists every fold file for a sweep
def cv_sweep_fold_files(name):
    return [f"{RESULTS}/cv/{name}/folds/{p}.parquet" for p in CVS[name].paramspace.instance_patterns]


wildcard_constraints:
    cv_sweep="|".join(CV_SWEEP_NAMES),
    cell=r"[^/]+",
    rep=r"\d+",  # same as rules/cv.smk; restated so this file does not rely on it


# --- stage 1 -------------------------------------------------------------------

# K=1 np fit to the twoSimple data of a (cell, rep)
rule cv_sweep_stage1_fit:
    output:
        f"{RESULTS}/cv/{{cv_sweep}}/stage1/{{cell}}_rep~{{rep}}.npz"
    params:
        model_spec=lambda wc: json.loads(
            CV_SWEEP_CELL_ROWS[wc.cv_sweep][(wc.cell, int(wc.rep))]["stage1_model_spec_json"]),
        dgp=lambda wc: build_sweep_sbm_dgp(CV_SWEEP_CELL_ROWS[wc.cv_sweep][(wc.cell, int(wc.rep))]),
    script:
        "../scripts/fit_model.py"

# outputs kernel text file
rule matched_kernel:
    input:
        f"{RESULTS}/cv/{{cv_sweep}}/stage1/{{cell}}_rep~{{rep}}.npz"
    output:
        f"{RESULTS}/cv/{{cv_sweep}}/kernels/{{cell}}_rep~{{rep}}.txt"
    script:
        "../scripts/make_matched_kernel.py"


# --- stage 2 -------------------------------------------------------------------

rule run_cv_sweep_task:
    # one job per (cell, DGP, K, rep, fold)
    input:
        kernel=cv_sweep_kernel_input,
    output:
        f"{RESULTS}/cv/{{cv_sweep}}/folds/{{params}}.parquet"
    params:
        dgp=cv_sweep_dgp,
        fixed=cv_sweep_row,
        axes=lambda wc: cv_sweep_axes(wc.cv_sweep),
    script:
        "../scripts/run_cv_sweep_task.py"


# --- stage 3 -------------------------------------------------------------------

rule aggregate_cv_sweep:
    input:
        lambda wc: cv_sweep_fold_files(wc.cv_sweep)
    output:
        f"{RESULTS}/cv/{{cv_sweep}}/_cv_curve.parquet",
        f"{RESULTS}/cv/{{cv_sweep}}/_cv_selection.parquet",
        f"{RESULTS}/cv/{{cv_sweep}}/_cv_by_cell.parquet",
    params:
        axes=lambda wc: cv_sweep_axes(wc.cv_sweep),
    script:
        "../scripts/aggregate_cv_sweep.py"
