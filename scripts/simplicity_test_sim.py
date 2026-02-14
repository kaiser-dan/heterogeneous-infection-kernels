"""
Simplicity Test: Simulate contagion on a 2-block SBM and fit Stan models.

Experiment 1 (null): All nodes share the same simple contagion kernel.
Experiment 2 (alt):  All nodes share a sigmoid (threshold) contagion kernel.
Experiment 3 (alt):  Two-block mixture of simple contagions (beta1 << beta2).

For each experiment we fit:
  - non_parametric_monotonic.stan (K=1)
  - model_simple_contagion.stan   (K=1)

Then run posterior predictive calibration (PPC) to build a null distribution
of the non-centrality parameter λ under the simple contagion hypothesis.

Posterior samples are saved to results/simplicity_test/.
"""

import sys
import numpy as np
import pandas as pd
import json
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cmdstanpy import CmdStanModel
from src.dynamics import heterogeneous_contagion_process, get_activations
from src.networks import sbm

# ── Configuration ───────────────────────────────────────────────────────────
SEED = 42
V = 300          # nodes (150 per block)
K_DEGREE = 14    # average degree
EPSILON = 0.7    # SBM mixing (modular blocks)
GAMMA = 0.15     # recovery rate (SIS dynamics)
TMAX = 200       # moderate run length
DT = 1

# Kernel parameters — Experiment 1 (simple contagion)
BETA = 0.08
N_INITIAL_INFECTED = 20

# Kernel parameters — Experiment 2 (moderate sigmoid)
RHO_L = 0.02           # low baseline
RHO_H = 0.50           # moderate saturation (NOT 1.0)
W = 1.5                # gentle slope
K0 = 4.0               # inflection at 4 infected neighbors

# Kernel parameters — Experiment 3 (two-block simple contagion mixture)
BETA_1 = 0.04          # Block 1: barely supercritical
BETA_2 = 0.20          # Block 2: strongly supercritical

# Stan sampling
ITER_SAMPLING = 500
ITER_WARMUP = 500
CHAINS = 2

# PPC
N_PPC_REPLICATES = 50

# Paths
RESULTS_DIR = PROJECT_ROOT / "results" / "simplicity_test"
STAN_DIR = PROJECT_ROOT / "src" / "models"

# ── Helpers ─────────────────────────────────────────────────────────────────

def simple_contagion_kernel(V, kmax, beta):
    """Build a V x kmax kernel matrix for simple contagion: c(nu) = 1 - (1-beta)^nu."""
    c = np.zeros((V, kmax))
    for nu in range(1, kmax + 1):
        c[:, nu - 1] = 1.0 - (1.0 - beta) ** nu
    return c


def sigmoid_kernel(V, kmax, rho_l, rho_h, w, k0):
    """Build a V x kmax kernel matrix for sigmoid/threshold contagion."""
    c = np.zeros((V, kmax))
    for nu in range(1, kmax + 1):
        c[:, nu - 1] = rho_l + (rho_h - rho_l) / (1.0 + np.exp(-w * (nu - k0)))
    return c


def run_experiment(name, A, c, kmax, seed):
    """Simulate contagion and extract M, N matrices."""
    np.random.seed(seed)
    x0 = np.zeros(A.shape[0])
    x0[np.random.choice(A.shape[0], N_INITIAL_INFECTED, replace=False)] = 1

    print(f"  Simulating {name} (V={A.shape[0]}, kmax={kmax}, tmax={TMAX})...")
    states = heterogeneous_contagion_process(A, GAMMA, c, x0, tmax=TMAX, dt=DT, random_seed=seed)
    M, N = get_activations(states.T, A, kmax)

    total_infections = M.sum()
    total_non = N.sum()
    print(f"  Total infection events: {total_infections:.0f}, non-infection events: {total_non:.0f}")
    if total_infections < 50:
        print(f"  WARNING: Very few infection events. Consider increasing TMAX or BETA.")

    return M, N


def fit_stan_model(stan_file, M, N, V, kmax, experiment_name, model_label, output_dir):
    """Fit a Stan model and save posterior samples."""
    print(f"  Fitting {model_label} to {experiment_name}...")
    model = CmdStanModel(stan_file=str(stan_file))

    stan_data = {
        "V": V,
        "kmax": kmax,
        "K": 1,
        "M": M.tolist(),
        "N": N.tolist(),
        "DEVIATION_SCALE": 0.1,
        "LOGIT_P_LOCATION": 0.0,
        "LOGIT_P_SCALE": 2.0,
    }

    # non_parametric_monotonic needs CONCENTRATION
    stan_code = stan_file.read_text()
    if "CONCENTRATION" in stan_code:
        stan_data["CONCENTRATION"] = 1.0

    fit = model.sample(
        data=stan_data,
        iter_sampling=ITER_SAMPLING,
        iter_warmup=ITER_WARMUP,
        chains=CHAINS,
        show_progress=True,
    )

    # Save
    out = output_dir / experiment_name / model_label
    out.mkdir(parents=True, exist_ok=True)

    fit.summary().to_parquet(out / "inference_summary.parquet")
    fit.draws_pd().to_parquet(out / "posterior_samples.parquet")
    print(f"  Saved to {out}")

    return fit


# ── PPC: Null distribution of λ ─────────────────────────────────────────────

def compute_lambda(mu_samples, rho_samples):
    """Compute non-centrality parameter λ from NP and SC posteriors."""
    k = mu_samples.shape[1]
    nu = np.arange(1, k + 1)

    mu_bar = mu_samples.mean(axis=0)
    Sigma = np.cov(mu_samples, rowvar=False)
    Sigma += 1e-8 * np.eye(k)
    L = np.linalg.cholesky(Sigma)

    rho_hat = np.mean(rho_samples)
    mu_sc = 1.0 - (1.0 - rho_hat) ** nu

    z_offset = np.linalg.solve(L, mu_bar - mu_sc)
    return float(z_offset @ z_offset)


def load_mu_samples(path):
    """Load NP kernel mode posterior samples from parquet."""
    draws = pd.read_parquet(path)
    mu_cols = sorted(
        [c for c in draws.columns if c.startswith("mu[1,")],
        key=lambda c: int(c.split(",")[1].rstrip("]")),
    )
    return draws[mu_cols].values


def load_rho_samples(path):
    """Load SC rho posterior samples from parquet."""
    draws = pd.read_parquet(path)
    return draws["rho[1]"].values


def run_ppc(experiment_name, rho_posterior, A, kmax, output_dir):
    """
    Posterior predictive calibration for the simplicity test.

    For each replicate:
      1. Draw ρ from the SC posterior fit to the real data
      2. Build a simple contagion kernel with that ρ
      3. Simulate SIS on the same network A
      4. Fit both NP and SC models to the simulated data
      5. Compute λ (non-centrality parameter)

    Returns array of λ values under the null.
    """
    V = A.shape[0]
    np_mono_stan = STAN_DIR / "non_parametric_monotonic.stan"
    simple_stan = STAN_DIR / "model_simple_contagion.stan"

    # Pre-compile models once
    np_model = CmdStanModel(stan_file=str(np_mono_stan))
    sc_model = CmdStanModel(stan_file=str(simple_stan))

    # Draw rho values for each replicate from the SC posterior
    rng = np.random.default_rng(SEED + 100)
    rho_draws = rng.choice(rho_posterior, size=N_PPC_REPLICATES, replace=True)

    ppc_dir = output_dir / experiment_name / "ppc"
    ppc_dir.mkdir(parents=True, exist_ok=True)

    lambdas_null = np.full(N_PPC_REPLICATES, np.nan)

    for r in range(N_PPC_REPLICATES):
        rho_r = rho_draws[r]
        seed_r = SEED + 1000 + r
        print(f"  PPC replicate {r+1}/{N_PPC_REPLICATES} (ρ={rho_r:.4f})...")

        # 1. Simulate under simple contagion with this ρ
        c_null = simple_contagion_kernel(V, kmax, rho_r)
        try:
            M_r, N_r = run_experiment(f"ppc_rep_{r}", A, c_null, kmax, seed=seed_r)
        except Exception as e:
            print(f"    Simulation failed: {e}")
            continue

        if M_r.sum() < 20:
            print(f"    Skipping: only {M_r.sum():.0f} infection events")
            continue

        # 2. Fit NP model
        stan_data = {
            "V": V, "kmax": kmax, "K": 1,
            "M": M_r.tolist(), "N": N_r.tolist(),
            "DEVIATION_SCALE": 0.1,
            "LOGIT_P_LOCATION": 0.0, "LOGIT_P_SCALE": 2.0,
            "CONCENTRATION": 1.0,
        }
        try:
            np_fit = np_model.sample(
                data=stan_data,
                iter_sampling=ITER_SAMPLING, iter_warmup=ITER_WARMUP,
                chains=CHAINS, show_progress=False,
            )
        except Exception as e:
            print(f"    NP fit failed: {e}")
            continue

        # 3. Fit SC model
        stan_data_sc = {k: v for k, v in stan_data.items() if k != "CONCENTRATION"}
        try:
            sc_fit = sc_model.sample(
                data=stan_data_sc,
                iter_sampling=ITER_SAMPLING, iter_warmup=ITER_WARMUP,
                chains=CHAINS, show_progress=False,
            )
        except Exception as e:
            print(f"    SC fit failed: {e}")
            continue

        # 4. Extract posteriors and compute λ
        np_draws = np_fit.draws_pd()
        mu_cols = sorted(
            [c for c in np_draws.columns if c.startswith("mu[1,")],
            key=lambda c: int(c.split(",")[1].rstrip("]")),
        )
        mu_samples_r = np_draws[mu_cols].values

        sc_draws = sc_fit.draws_pd()
        rho_samples_r = sc_draws["rho[1]"].values

        lambdas_null[r] = compute_lambda(mu_samples_r, rho_samples_r)
        print(f"    λ = {lambdas_null[r]:.1f}")

    # Save
    valid = lambdas_null[~np.isnan(lambdas_null)]
    np.save(ppc_dir / "lambdas_null.npy", valid)
    print(f"  PPC complete: {len(valid)}/{N_PPC_REPLICATES} valid replicates")
    print(f"  λ_null: median={np.median(valid):.1f}, "
          f"mean={np.mean(valid):.1f}, 95th={np.percentile(valid, 95):.1f}")

    return valid


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Simplicity Test: SBM Simulation + Stan Fitting")
    print("=" * 60)

    # 1. Generate network
    print("\n[1/5] Generating 2-block SBM network...")
    A = sbm(V, K_DEGREE, EPSILON, seed=SEED)
    kmax = int(np.max(A.sum(axis=1)))
    block_labels = np.array([0] * (V // 2) + [1] * (V // 2))
    print(f"  V={V}, kmax={kmax}, avg_degree={A.sum(axis=1).mean():.1f}")

    # Save network metadata
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "V": V,
        "kmax": kmax,
        "K_DEGREE": K_DEGREE,
        "EPSILON": EPSILON,
        "GAMMA": GAMMA,
        "TMAX": TMAX,
        "BETA": BETA,
        "RHO_L": RHO_L,
        "RHO_H": RHO_H,
        "W": W,
        "K0": K0,
        "BETA_1": BETA_1,
        "BETA_2": BETA_2,
        "SEED": SEED,
        "ITER_SAMPLING": ITER_SAMPLING,
        "ITER_WARMUP": ITER_WARMUP,
        "CHAINS": CHAINS,
    }
    with open(RESULTS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save adjacency and block labels
    np.save(RESULTS_DIR / "adjacency.npy", A)
    np.save(RESULTS_DIR / "block_labels.npy", block_labels)

    # 2. Experiment 1: Homogeneous simple contagion
    print("\n[2/5] Experiment 1: Homogeneous (all nodes beta={})".format(BETA))
    c_homo = simple_contagion_kernel(V, kmax, BETA)
    M1, N1 = run_experiment("experiment_1_homogeneous", A, c_homo, kmax, seed=SEED)
    np.save(RESULTS_DIR / "M1.npy", M1)
    np.save(RESULTS_DIR / "N1.npy", N1)

    # 3. Experiment 2: Sigmoid / threshold contagion
    print(f"\n[3/5] Experiment 2: Sigmoid (rho_l={RHO_L}, rho_h={RHO_H}, w={W}, k0={K0})")
    c_sigmoid = sigmoid_kernel(V, kmax, RHO_L, RHO_H, W, K0)
    M2, N2 = run_experiment("experiment_2_sigmoid", A, c_sigmoid, kmax, seed=SEED + 1)
    np.save(RESULTS_DIR / "M2.npy", M2)
    np.save(RESULTS_DIR / "N2.npy", N2)

    # 4. Experiment 3: Two-block mixture of simple contagions
    print(f"\n[4/5] Experiment 3: Two-block mixture (beta1={BETA_1}, beta2={BETA_2})")
    c_mixture = np.zeros((V, kmax))
    c_mixture[: V // 2, :] = simple_contagion_kernel(V // 2, kmax, BETA_1)
    c_mixture[V // 2 :, :] = simple_contagion_kernel(V // 2, kmax, BETA_2)
    M3, N3 = run_experiment("experiment_3_mixture", A, c_mixture, kmax, seed=SEED + 2)
    np.save(RESULTS_DIR / "M3.npy", M3)
    np.save(RESULTS_DIR / "N3.npy", N3)

    # 5. Fit models
    print("\n[5/5] Fitting Stan models...")
    np_mono_stan = STAN_DIR / "non_parametric_monotonic.stan"
    simple_stan = STAN_DIR / "model_simple_contagion.stan"

    # Experiment 1 x 2 models
    fit_stan_model(np_mono_stan, M1, N1, V, kmax, "experiment_1_homogeneous", "non_parametric_monotonic", RESULTS_DIR)
    fit_stan_model(simple_stan, M1, N1, V, kmax, "experiment_1_homogeneous", "simple_contagion", RESULTS_DIR)

    # Experiment 2 x 2 models
    fit_stan_model(np_mono_stan, M2, N2, V, kmax, "experiment_2_sigmoid", "non_parametric_monotonic", RESULTS_DIR)
    fit_stan_model(simple_stan, M2, N2, V, kmax, "experiment_2_sigmoid", "simple_contagion", RESULTS_DIR)

    # Experiment 3 x 2 models
    fit_stan_model(np_mono_stan, M3, N3, V, kmax, "experiment_3_mixture", "non_parametric_monotonic", RESULTS_DIR)
    fit_stan_model(simple_stan, M3, N3, V, kmax, "experiment_3_mixture", "simple_contagion", RESULTS_DIR)

    # 6. Posterior Predictive Calibration
    #    Build null distribution of λ for each experiment by:
    #    drawing ρ from that experiment's SC posterior → simulate → refit → compute λ
    print("\n[6/6] Posterior Predictive Calibration...")

    for exp_name in ["experiment_1_homogeneous", "experiment_2_sigmoid", "experiment_3_mixture"]:
        print(f"\n  --- PPC for {exp_name} ---")
        rho_post = load_rho_samples(
            RESULTS_DIR / exp_name / "simple_contagion" / "posterior_samples.parquet"
        )
        run_ppc(exp_name, rho_post, A, kmax, RESULTS_DIR)

    print("\n" + "=" * 60)
    print("Done! Results in:", RESULTS_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
