"""
Generate synthetic SBM+SIS contagion datasets for inference pipeline.
Called by workflow/rules/simulation.smk rule run_complexity.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.simulation.dynamics import get_activations, heterogeneous_contagion_process
from src.simulation.utils import (
    assign_kernels,
    build_graph,
    build_initial_state,
    build_kernel_bank,
)

def save_outputs(
    output_dir: Path,
    adjacency: np.ndarray,
    states: np.ndarray,
    M: np.ndarray,
    N: np.ndarray,
    kernel_ids: np.ndarray,
    kernel_matrix: np.ndarray,
    metadata: dict,
) -> None:
    """Write all dataset artifacts to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # M and N as .npy — inference.py loads these directly with np.load
    np.save(output_dir / "M.npy", M)
    np.save(output_dir / "N.npy", N)

    # Supporting parquets for inspection / figure scripts
    pd.DataFrame(states).to_parquet(output_dir / "exposures.parquet", index=False)

    src, dst = np.where(np.triu(adjacency, k=1) > 0)
    pd.DataFrame({
        "src":    src.astype(int),
        "dst":    dst.astype(int),
        "weight": adjacency[src, dst].astype(float),
    }).to_parquet(output_dir / "edges.parquet", index=False)

    kernel_cols = [f"nu_{i}" for i in range(1, kernel_matrix.shape[1] + 1)]
    kernels_df  = pd.DataFrame(kernel_matrix, columns=kernel_cols)
    kernels_df.insert(0, "kernel_id", kernel_ids.astype(int))
    kernels_df.insert(0, "node_id", np.arange(kernel_matrix.shape[0], dtype=int))
    kernels_df.to_parquet(output_dir / "kernels.parquet", index=False)

    pd.DataFrame([metadata]).to_parquet(output_dir / "metadata.parquet", index=False)
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base in place."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def main(
    config_path: Path,
    output_dir: Path,
    data_name: str,
    synthetic_overrides: dict | None = None,
) -> None:
    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    synth = dict(config.get("synthetic", {}))
    if synthetic_overrides:
        synth = _deep_merge(synth, synthetic_overrides)

    graph_cfg      = synth.get("graph", {})
    contagion_cfg  = synth.get("contagion", {})
    assignment_cfg = contagion_cfg.get("assignment", {"mode": "iid"})
    kernels_cfg    = contagion_cfg.get("kernels", [])
    if not kernels_cfg:
        raise ValueError("synthetic.contagion.kernels must contain at least one kernel definition.")

    seed  = int(synth.get("seed", 37))
    gamma = float(contagion_cfg.get("gamma", 0.1))
    tmax  = int(contagion_cfg.get("tmax", 500))
    dt    = float(contagion_cfg.get("dt", 1.0))

    adjacency = build_graph(graph_cfg=graph_cfg, seed=seed)
    n         = adjacency.shape[0]
    kmax      = max(int(synth.get("kmax", 0)), int(np.max(adjacency.sum(axis=1))))

    kernel_bank   = build_kernel_bank(kernels_cfg=kernels_cfg, kmax=kmax)
    kernel_ids    = assign_kernels(
        n=n,
        graph_type=graph_cfg.get("type", "sbm"),
        assignment_cfg=assignment_cfg,
        n_kernels=len(kernel_bank),
        seed=seed,
    )
    kernel_matrix = np.vstack([kernel_bank[idx] for idx in kernel_ids])

    x0     = build_initial_state(n=n, initial_infected=contagion_cfg.get("initial_infected", max(1, n // 10)), seed=seed)
    states = heterogeneous_contagion_process(
        adjacency, gamma=gamma, c=kernel_matrix, x0=x0,
        tmin=0, tmax=tmax, dt=dt, random_seed=seed,
    )
    M, N = get_activations(states.T, adjacency, kmax)

    metadata = {
        "data_name":             data_name,
        "schema_version":        "1",
        "source":                "synthetic",
        "graph_type":            graph_cfg.get("type", "sbm"),
        "graph_params_json":     json.dumps(graph_cfg.get("params", {}), sort_keys=True),
        "assignment_mode":       assignment_cfg.get("mode", "iid"),
        "assignment_params_json": json.dumps(
            {k: v for k, v in assignment_cfg.items() if k != "mode"}, sort_keys=True
        ),
        "kernels_json":          json.dumps(kernels_cfg, sort_keys=True),
        "n":                     n,
        "kmax":                  kmax,
        "gamma":                 gamma,
        "tmax":                  tmax,
        "dt":                    dt,
        "seed":                  seed,
        "initial_infected":      float(x0.sum()),
    }

    save_outputs(
        output_dir=output_dir,
        adjacency=adjacency,
        states=states,
        M=M,
        N=N,
        kernel_ids=kernel_ids,
        kernel_matrix=kernel_matrix,
        metadata=metadata,
    )
    print(f"Saved synthetic dataset '{data_name}' to {output_dir}  (n={n}, kmax={kmax})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic contagion dataset.")
    parser.add_argument("--config",      type=Path, required=True)
    parser.add_argument("--output-dir",  type=Path, required=True)
    parser.add_argument("--data-name",   type=str,  required=True)
    parser.add_argument("--synthetic-overrides", type=str, default="{}")
    args = parser.parse_args()

    main(
        config_path=args.config,
        output_dir=args.output_dir,
        data_name=args.data_name,
        synthetic_overrides=json.loads(args.synthetic_overrides),
    )
