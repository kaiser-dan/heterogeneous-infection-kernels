"""PPC rules: for each omega in the sweep, one observed fit, 
then NREP independently parallelizable null fits (wildcard r), then a per-omega summary.
"""

from params_io import load_experiment

PARAMS_DIR = ROOT / "config" / "params"

PPC = load_experiment("ppc", PARAMS_DIR)
NREP = int(PPC.frame.iloc[0]["nrep"]) # constant across the frame

# per-omega fixed params + the r-values that belong to each omega
FIXED_BY_OMEGA, R_BY_OMEGA = {}, {}
for om, sub in PPC.frame.groupby("omega", sort=True):
    tok = str(om)
    FIXED_BY_OMEGA[tok] = sub.iloc[0].to_dict()   # constants + this omega
    R_BY_OMEGA[tok] = list(sub["r"])

PPC_OMEGAS = list(FIXED_BY_OMEGA) # exported for `rule all` in the Snakefile

wildcard_constraints:
    r = r"\d+",
    omega = r"[0-9.eE+-]+",


rule ppc_observed:
    output:
        f"{RESULTS}/ppc/observed/omega~{{omega}}.npz"
    params:
        fixed=lambda wc: FIXED_BY_OMEGA[wc.omega],
    script:
        "../scripts/run_ppc_observed.py"

rule ppc_null:
    input:
        f"{RESULTS}/ppc/observed/omega~{{omega}}.npz"
    output:
        f"{RESULTS}/ppc/nulls/{PPC.paramspace.wildcard_pattern}.npz"
    params:
        fixed=lambda wc: PPC.row(wc),
    script:
        "../scripts/run_ppc_null.py"

rule ppc_summary:
    input:
        observed=f"{RESULTS}/ppc/observed/omega~{{omega}}.npz",
        nulls=lambda wc: expand(
            f"{RESULTS}/ppc/nulls/omega~{wc.omega}/r~{{r}}.npz",
            r=R_BY_OMEGA[wc.omega],
        ),
    output:
        result_json=f"{RESULTS}/ppc/omega~{{omega}}/ppc_result.json",
        null_dist=f"{RESULTS}/ppc/omega~{{omega}}/ppc_null_distribution.npz",
    params:
        fixed=lambda wc: FIXED_BY_OMEGA[wc.omega],
    script:
        "../scripts/ppc_summary.py"
