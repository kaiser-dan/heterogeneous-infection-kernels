"""Assemble full DGP dicts from a resolved parameter-frame row."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))

def build_sweep_sbm_dgp(row: dict) -> dict:
    """Full dgp dict for a 2-block sbm SBM, given one resolved row 
    (fixed columns + wildcard values already merged by params_io.ExperimentFrame.row).
    """
    return {
        "gamma": row["gamma"],
        "rho": row["rho"],
        "tmax": row["tmax"],
        "rep": int(row["rep"]),
        "graph": {"type": "sbm", "params": {
            "n": int(row["graph_n"]),
            "avg_degree": int(row["graph_avg_degree"]),
            "epsilon": row["epsilon"],
            "delta": row["delta"],
        }},
        "kernels": [
            {"type": "simple", "beta": row["beta1"]},
            {"type": "simple", "beta": row["beta2"]},
        ],
        "assignment": {"mode": "block_flip", "omega": row["omega"]},
    }
    

def build_sweep_edgelist_dgp(row: dict) -> dict:
    """Full dgp dict for Congress network, given one resolved row
    (fixed columns + wildcard values already merged by params_io.ExperimentFrame.row).
    """
    
    communities = np.loadtxt(row["communities_fpath"], dtype=int)
    
    return {
        "gamma": row["gamma"],
        "rho": row["rho"],
        "tmax": row["tmax"],
        "rep": int(row["rep"]),
        "graph": {"type": "edgelist", "params": {"filename": row["graph_fpath"]}},
        "kernels": [
            {"type": "simple", "beta": row["beta1"]},
            {"type": "simple", "beta": row["beta2"]}
        ],
        "assignment": {"mode": "block_flip", "omega": row["omega"], "communities": communities},
    }


def cv_kernel_files(kernels: list[dict]) -> list[str]:
    """values_file paths (relative to repo root) used by nonparametric kernels."""
    return [k["values_file"] for k in kernels if k.get("type") == "nonparametric"]


def build_cv_dgp(row: dict, root: str | Path = ".") -> dict:
    """Full dgp dict for one CV task, given one resolved row (fixed columns
    + wildcard values already merged by params_io.ExperimentFrame.row). 
    `row["kernels_json"]`/`assignment_json` are that row's dgp_label's family, already resolved by the frame.
    """
    kernels = [dict(k) for k in json.loads(row["kernels_json"])]
    for k in kernels:
        if k.get("type") == "nonparametric":
            k["values_file"] = str(Path(root) / k["values_file"])

    return {
        "gamma": row["gamma"],
        "rho": row["rho"],
        "tmax": row["tmax"],
        "rep": int(row["rep"]),
        "graph": {"type": row["graph_type"], "params": {
            "n": int(row["graph_n"]),
            "avg_degree": int(row["graph_avg_degree"]),
            "epsilon": row["graph_epsilon"],
            "delta": row["graph_delta"],
        }},
        "kernels": kernels,
        "assignment": json.loads(row["assignment_json"]),
    }
