import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dynamics import get_activations, heterogeneous_contagion_process
from src.networks import er, sbm, zkc


def simple_kernel(beta: float, kmax: int) -> np.ndarray:
    nu = np.arange(1, kmax + 1, dtype=float)
    return 1.0 - (1.0 - beta) ** nu


def sigmoid_kernel(rho_l: float, rho_h: float, w: float, k0: float, kmax: int) -> np.ndarray:
    nu = np.arange(1, kmax + 1, dtype=float)
    return rho_l + (rho_h - rho_l) / (1.0 + np.exp(-w * (nu - k0)))


def nonparametric_kernel(values: list[float], kmax: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size < kmax:
        raise ValueError(f"Nonparametric kernel length {arr.size} is smaller than kmax={kmax}.")
    return arr[:kmax]


def power_law_kernel(lam: float, alpha: float, kmax: int) -> np.ndarray:
    nu = np.arange(1, kmax + 1, dtype=float)
    return np.clip(lam * nu**alpha, 0.0, 1.0)



def build_graph(graph_cfg: dict, seed: int) -> np.ndarray:
    graph_type = graph_cfg.get("type", "sbm")
    params = graph_cfg.get("params", {})

    if graph_type == "sbm":
        n = int(params["n"])
        k = float(params.get("avg_degree", params.get("k")))
        epsilon = float(params["epsilon"])
        phi = float(params.get("phi", 0.5))
        return np.asarray(sbm(n=n, k=k, epsilon=epsilon, phi=phi, seed=seed), dtype=int)
    if graph_type == "er":
        n = int(params["n"])
        p = params.get("p")
        avg_degree = params.get("avg_degree")
        return np.asarray(er(n=n, p=p, k=avg_degree, seed=seed), dtype=int)
    if graph_type == "zkc":
        return np.asarray(zkc(), dtype=int)

    raise ValueError(f"Unsupported graph type: {graph_type}")


def build_kernel_bank(kernels_cfg: list[dict], kmax: int) -> list[np.ndarray]:
    bank = []
    for kernel_cfg in kernels_cfg:
        ktype = kernel_cfg.get("type")
        if ktype == "simple":
            bank.append(simple_kernel(beta=float(kernel_cfg["beta"]), kmax=kmax))
        elif ktype == "sigmoid":
            bank.append(
                sigmoid_kernel(
                    rho_l=float(kernel_cfg["rho_l"]),
                    rho_h=float(kernel_cfg["rho_h"]),
                    w=float(kernel_cfg["w"]),
                    k0=float(kernel_cfg["k0"]),
                    kmax=kmax,
                )
            )
        elif ktype == "nonparametric":
            bank.append(nonparametric_kernel(values=kernel_cfg["values"], kmax=kmax))
        elif ktype == "power_law":
            bank.append(
                power_law_kernel(
                    lam=float(kernel_cfg["lam"]),
                    alpha=float(kernel_cfg["alpha"]),
                    kmax=kmax,
                )
            )
        else:
            raise ValueError(f"Unsupported kernel type: {ktype}")
    return bank


def assign_kernels(
    n: int,
    graph_type: str,
    assignment_cfg: dict,
    n_kernels: int,
    seed: int,
) -> np.ndarray:
    mode = assignment_cfg.get("mode", "block_flip")
    rng = np.random.default_rng(seed + 17)

    if mode == "block_flip":
        if n_kernels != 2:
            raise ValueError("block_flip assignment currently requires exactly 2 kernels.")
        if graph_type != "sbm":
            raise ValueError("block_flip assignment is currently only supported for SBM.")

        omega = float(assignment_cfg.get("omega", 0.0))
        split = n // 2
        labels = np.zeros(n, dtype=int)
        labels[split:] = 1
        flips = rng.random(n) < omega
        labels[flips] = 1 - labels[flips]
        return labels

    if mode == "iid":
        probs = assignment_cfg.get("probs")
        if probs is None:
            probs = np.ones(n_kernels, dtype=float) / float(n_kernels)
        else:
            probs = np.asarray(probs, dtype=float).reshape(-1)
            if probs.size != n_kernels:
                raise ValueError("Length of assignment probs must match number of kernels.")
            probs = probs / probs.sum()
        return rng.choice(n_kernels, size=n, p=probs).astype(int)

    if mode == "explicit":
        labels = np.asarray(assignment_cfg.get("labels", []), dtype=int).reshape(-1)
        if labels.size != n:
            raise ValueError("Explicit assignment requires one label per node.")
        return labels

    raise ValueError(f"Unsupported assignment mode: {mode}")


def build_initial_state(n: int, initial_infected, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 23)
    if isinstance(initial_infected, float) and 0 < initial_infected <= 1:
        n_init = max(1, int(round(initial_infected * n)))
    else:
        n_init = int(initial_infected)
    n_init = max(1, min(n_init, n))

    x0 = np.zeros(n, dtype=int)
    seeds = rng.choice(n, size=n_init, replace=False)
    x0[seeds] = 1
    return x0


def save_outputs(
    output_dir: Path,
    adjacency: np.ndarray,
    states: np.ndarray,
    M: np.ndarray,
    N: np.ndarray,
    kernel_ids: np.ndarray,
    kernel_matrix: np.ndarray,
    metadata: dict,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(M).to_parquet(output_dir / "M.parquet", index=False)
    pd.DataFrame(N).to_parquet(output_dir / "N.parquet", index=False)
    pd.DataFrame(states).to_parquet(output_dir / "exposures.parquet", index=False)

    src, dst = np.where(np.triu(adjacency, k=1) > 0)
    edges = pd.DataFrame(
        {
            "src": src.astype(int),
            "dst": dst.astype(int),
            "weight": adjacency[src, dst].astype(float),
        }
    )
    edges.to_parquet(output_dir / "edges.parquet", index=False)

    kernel_cols = [f"nu_{i}" for i in range(1, kernel_matrix.shape[1] + 1)]
    kernels_df = pd.DataFrame(kernel_matrix, columns=kernel_cols)
    kernels_df.insert(0, "kernel_id", kernel_ids.astype(int))
    kernels_df.insert(0, "node_id", np.arange(kernel_matrix.shape[0], dtype=int))
    kernels_df.to_parquet(output_dir / "kernels.parquet", index=False)

    pd.DataFrame([metadata]).to_parquet(output_dir / "metadata.parquet", index=False)
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True)


def deep_merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def main(config_path: Path, output_dir: Path, data_name: str, synthetic_overrides: dict | None = None):
    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    if config.get("source") != "synthetic":
        raise ValueError("Synthetic generation requires source: synthetic in config.")

    synth = config.get("synthetic", {})
    if synthetic_overrides:
        synth = deep_merge(dict(synth), synthetic_overrides)
    graph_cfg = synth.get("graph", {})
    contagion_cfg = synth.get("contagion", {})

    seed = int(synth.get("seed", 37))
    gamma = float(contagion_cfg.get("gamma", 0.1))
    tmax = int(contagion_cfg.get("tmax", 500))
    dt = float(contagion_cfg.get("dt", 1))

    adjacency = build_graph(graph_cfg=graph_cfg, seed=seed)
    n = int(adjacency.shape[0])
    observed_kmax = int(np.max(adjacency.sum(axis=1)))
    requested_kmax = int(synth.get("kmax", observed_kmax))
    kmax = max(requested_kmax, observed_kmax)

    kernels_cfg = contagion_cfg.get("kernels", [])
    if not kernels_cfg:
        raise ValueError("synthetic.contagion.kernels must contain at least one kernel definition.")

    kernel_bank = build_kernel_bank(kernels_cfg=kernels_cfg, kmax=kmax)
    assignment_cfg = contagion_cfg.get("assignment", {"mode": "iid"})
    kernel_ids = assign_kernels(
        n=n,
        graph_type=graph_cfg.get("type", "sbm"),
        assignment_cfg=assignment_cfg,
        n_kernels=len(kernel_bank),
        seed=seed,
    )
    kernel_matrix = np.vstack([kernel_bank[idx] for idx in kernel_ids])

    initial_infected = contagion_cfg.get("initial_infected", max(1, n // 10))
    x0 = build_initial_state(n=n, initial_infected=initial_infected, seed=seed)

    states = heterogeneous_contagion_process(
        adjacency,
        gamma=gamma,
        c=kernel_matrix,
        x0=x0,
        tmin=0,
        tmax=tmax,
        dt=dt,
        random_seed=seed,
    )
    M, N = get_activations(states.T, adjacency, kmax)

    metadata = {
        "data_name": data_name,
        "schema_version": "1",
        "source": "synthetic",
        "graph_type": graph_cfg.get("type", "sbm"),
        "graph_params_json": json.dumps(graph_cfg.get("params", {}), sort_keys=True),
        "assignment_mode": assignment_cfg.get("mode", "iid"),
        "assignment_params_json": json.dumps(
            {k: v for k, v in assignment_cfg.items() if k != "mode"},
            sort_keys=True,
        ),
        "kernels_json": json.dumps(kernels_cfg, sort_keys=True),
        "n": n,
        "kmax": kmax,
        "gamma": gamma,
        "tmax": tmax,
        "dt": dt,
        "seed": seed,
        "initial_infected": float(initial_infected)
        if isinstance(initial_infected, float)
        else int(initial_infected),
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic contagion data artifacts.")
    parser.add_argument("--config", type=Path, required=True, help="Path to sweep config YAML.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output synthetic data directory.")
    parser.add_argument("--data-name", type=str, required=True, help="Synthetic dataset name.")
    parser.add_argument(
        "--synthetic-overrides",
        type=str,
        default="{}",
        help="JSON object of nested overrides to apply under synthetic.*",
    )
    args = parser.parse_args()

    overrides = json.loads(args.synthetic_overrides)
    main(
        config_path=args.config,
        output_dir=args.output_dir,
        data_name=args.data_name,
        synthetic_overrides=overrides,
    )
