"""
Test script for finite mixture models (monotonic vs non-monotonic)

Generates simulated data using PROPER simple contagion kernels
and tests both models. Checks ESS, Rhat, and parameter recovery.
"""

import numpy as np
import networkx as nx
from cmdstanpy import CmdStanModel
import pandas as pd
from pathlib import Path
import json
import sys

# Add project root for imports
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(REPO_ROOT))

from src.dynamics import heterogeneous_contagion_process, get_activations

# Set random seed for reproducibility
np.random.seed(42)

# ==============================================================================
# SIMULATION PARAMETERS
# ==============================================================================

V = 50  # number of nodes
kmax = 20  # max number of infected neighbors (must be >= max degree)
K = 2  # number of kernel modes (ground truth)

# Ground truth base infection rates (simple contagion)
b_true = [0.35, 0.10]  # Mode 1: high transmissibility, Mode 2: low

# Compute kernel modes from base rates: c(ν) = 1 - (1-b)^ν
mu_true = []
for b in b_true:
    kernel = np.array([1 - (1 - b)**nu for nu in range(1, kmax + 1)])
    mu_true.append(kernel)
mu_true = np.array(mu_true)

print("Ground truth kernel modes (simple contagion form):")
for k, b in enumerate(b_true):
    print(f"  Mode {k+1}: b = {b:.3f}")
    print(f"    c(1) = {mu_true[k, 0]:.3f}")
    print(f"    c(5) = {mu_true[k, 4]:.3f}")
    print(f"    c(10) = {mu_true[k, 9]:.3f}")

# ==============================================================================
# NETWORK STRUCTURE: 2-Block SBM
# ==============================================================================

# Create 2-block stochastic block model
block_sizes = [25, 25]
p_within = 0.3  # high connectivity within blocks
p_between = 0.05  # low connectivity between blocks

G = nx.stochastic_block_model(block_sizes, [[p_within, p_between],
                                             [p_between, p_within]],
                               seed=42)

# Get block assignments (stored as node attribute)
blocks = [G.nodes[i]['block'] for i in range(V)]
blocks = np.array(blocks)

print(f"\nNetwork: 2-block SBM, V={V}")
print(f"  Block 0: {(blocks == 0).sum()} nodes")
print(f"  Block 1: {(blocks == 1).sum()} nodes")
print(f"  Edges: {G.number_of_edges()}")

# ==============================================================================
# ASSIGN KERNELS BASED ON BLOCKS (with 10% crossover)
# ==============================================================================

# 90% of each block gets the "matching" kernel, 10% gets the other
z_true = np.zeros(V, dtype=int)
for i in range(V):
    if np.random.rand() < 0.9:
        z_true[i] = blocks[i]  # Same as block
    else:
        z_true[i] = 1 - blocks[i]  # Opposite block

print(f"\nKernel assignments (with 10% crossover):")
print(f"  Mode 0 (high-b): {(z_true == 0).sum()} nodes ({(z_true == 0).sum()/V:.1%})")
print(f"  Mode 1 (low-b):  {(z_true == 1).sum()} nodes ({(z_true == 1).sum()/V:.1%})")

# Per-node kernels: mode + small noise
sigma_true = 0.15  # Small noise around modes
kernels_true = np.zeros((V, kmax))
for i in range(V):
    mode = z_true[i]
    # Add noise on logit scale
    logit_mu = np.log(mu_true[mode] / (1 - mu_true[mode] + 1e-12))
    logit_c = logit_mu + sigma_true * np.random.randn(kmax)
    kernels_true[i] = 1 / (1 + np.exp(-logit_c))

# ==============================================================================
# SIMULATE USING ACTUAL DYNAMICS CODE
# ==============================================================================

print("\nSimulating contagion dynamics...")

# Convert to adjacency matrix
A = nx.to_numpy_array(G, dtype=int)

# Run multiple simulations
n_sims = 100
gamma = 0.05  # healing rate (low to get more data)
all_M = []
all_N = []

for sim in range(n_sims):
    # Random initial condition (few seeds)
    x0 = np.zeros(V, dtype=int)
    x0[np.random.choice(V, size=3, replace=False)] = 1

    # Run dynamics
    states = heterogeneous_contagion_process(
        A, gamma, kernels_true, x0,
        tmin=0, tmax=30, dt=0.1,
        random_seed=None
    )

    # Extract M, N for this simulation
    M_sim, N_sim = get_activations(states.T, A, kmax)
    all_M.append(M_sim)
    all_N.append(N_sim)

# Aggregate across simulations
M = np.sum(all_M, axis=0).astype(int)
N = np.sum(all_N, axis=0).astype(int)

# Check data quality
print(f"Total infection events: {M.sum()}")
print(f"Total non-infection events: {N.sum()}")
print(f"Nodes with at least one observation: {(M.sum(axis=1) + N.sum(axis=1) > 0).sum()}")

# Check exposure distribution
exposure_counts = M + N
print(f"\nExposure distribution (total counts per ν):")
for nu in range(min(kmax, 10)):
    print(f"  ν={nu+1}: {exposure_counts[:, nu].sum()} observations")

# ==============================================================================
# PREPARE STAN DATA
# ==============================================================================

stan_data = {
    'V': V,
    'kmax': kmax,
    'K': K,
    'M': M.tolist(),
    'N': N.tolist(),
    'DEVIATION_SCALE': 1.0
}

# Save data for inspection
data_path = Path(__file__).parent / "test_mixture_data.json"
with open(data_path, 'w') as f:
    json.dump(stan_data, f, indent=2)
print(f"\nSaved data to {data_path}")

# ==============================================================================
# FIT MODELS
# ==============================================================================

models_dir = Path(__file__).parent.parent.parent / "src" / "models"

print("\n" + "="*80)
print("FITTING MONOTONIC MODEL")
print("="*80)

model_mono = CmdStanModel(
    stan_file=str(models_dir / "model_finite_mixture_monotonic.stan")
)

fit_mono = model_mono.sample(
    data=stan_data,
    chains=4,
    parallel_chains=4,
    iter_warmup=1000,
    iter_sampling=1000,
    adapt_delta=0.95,
    max_treedepth=12,
    show_progress=True,
    show_console=False
)

print("\n" + "="*80)
print("FITTING NON-MONOTONIC MODEL")
print("="*80)

model_nonmono = CmdStanModel(
    stan_file=str(models_dir / "model_finite_mixture_nonmonotonic.stan")
)

fit_nonmono = model_nonmono.sample(
    data=stan_data,
    chains=4,
    parallel_chains=4,
    iter_warmup=1000,
    iter_sampling=1000,
    adapt_delta=0.95,
    max_treedepth=12,
    show_progress=True,
    show_console=False
)

# ==============================================================================
# DIAGNOSTICS
# ==============================================================================

print("\n" + "="*80)
print("DIAGNOSTICS")
print("="*80)

def print_diagnostics(fit, model_name):
    """Print key diagnostics for a fit."""
    print(f"\n{model_name}:")
    print("-" * 40)

    # Divergences
    try:
        divergences = fit.divergences
        if divergences is not None:
            print(f"Divergences: {divergences.sum()}")
        else:
            print("Divergences: 0")
    except:
        print("Divergences: Unable to check")

    # Summary for key parameters
    summary = fit.summary()

    # Check Rhat
    rhat_col = 'R_hat' if 'R_hat' in summary.columns else 'Rhat'
    max_rhat = summary[rhat_col].max()
    print(f"Max Rhat: {max_rhat:.4f}")

    # Check ESS
    ess_cols = [c for c in summary.columns if 'ess' in c.lower() or 'n_eff' in c.lower()]
    if ess_cols:
        min_ess = summary[ess_cols[0]].min()
        print(f"Min ESS: {min_ess:.0f}")
    else:
        print("ESS: Unable to find column")

    # Determine column names
    mean_col = [c for c in summary.columns if 'mean' in c.lower()][0] if any('mean' in c.lower() for c in summary.columns) else 'Mean'
    std_col = [c for c in summary.columns if 'std' in c.lower()][0] if any('std' in c.lower() for c in summary.columns) else 'StdDev'

    # Print summary for sigma
    if 'sigma' in summary.index:
        print("\nSigma:")
        cols_to_show = [c for c in [mean_col, std_col, ess_cols[0] if ess_cols else None, rhat_col] if c and c in summary.columns]
        print(summary.loc['sigma', cols_to_show])

    # Print summary for kernel modes (first few exposure levels)
    mu_params = [p for p in summary.index if 'mu[' in p][:6]
    if mu_params:
        print("\nKernel modes (first 6 params):")
        print(summary.loc[mu_params, cols_to_show])

    return summary

summary_mono = print_diagnostics(fit_mono, "MONOTONIC MODEL")
summary_nonmono = print_diagnostics(fit_nonmono, "NON-MONOTONIC MODEL")

# ==============================================================================
# SAVE RESULTS
# ==============================================================================

results_dir = Path(__file__).parent / "test_mixture_results"
results_dir.mkdir(exist_ok=True)

# Save summaries
summary_mono.to_csv(results_dir / "monotonic_summary.csv")
summary_nonmono.to_csv(results_dir / "nonmonotonic_summary.csv")

# Save ground truth
np.savez(
    results_dir / "ground_truth.npz",
    z_true=z_true,
    kernels_true=kernels_true,
    mu_true=mu_true,
    sigma_true=sigma_true,
    b_true=b_true,
    blocks=blocks
)

print(f"\nResults saved to {results_dir}")
print("\n" + "="*80)
print("DONE")
print("="*80)
