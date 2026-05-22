import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import complexity measures
from src.complexity_measures.malahbois import compute_malahbois_complexity, simple_contagion_reference


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_metadata(data_dir: Path) -> pd.Series:
    """Load and parse metadata.parquet for a synthetic dataset."""
    meta_path = data_dir / "metadata.parquet"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    df = pd.read_parquet(meta_path)
    row = df.iloc[0]

    # Parse JSON fields
    graph_params = json.loads(row["graph_params_json"])
    assign_params = json.loads(row["assignment_params_json"])

    return pd.Series({
        "data_name": row["data_name"],
        "epsilon": float(graph_params.get("epsilon", np.nan)),
        "omega": float(assign_params.get("omega", np.nan)),
        "avg_degree": float(graph_params.get("avg_degree", np.nan)),
        "kmax": int(row.get("kmax", 20)),
        "N_nodes": int(row["n"]),
        "gamma": float(row["gamma"]),
    })


def load_simple_posterior(results_dir: Path) -> np.ndarray:
    """Load simple contagion posterior samples and extract rho."""
    post_path = results_dir / "stan_file~model_simple_contagion/posterior_samples.parquet"
    if not post_path.exists():
        # Try legacy path or other variations if needed, but for now strict
        raise FileNotFoundError(f"Simple posterior not found: {post_path}")

    # We only need rho[1] (or just rho if strict scalar)
    # The clean column name is typically 'rho[1]' or 'rho'
    cols = ["rho[1]", "rho"]
    df = pd.read_parquet(post_path)
    found_col = next((c for c in cols if c in df.columns), None)
    
    if not found_col:
        raise ValueError(f"Could not find rho column in {post_path}. Columns: {df.columns.tolist()[:5]}...")
        
    return df[found_col].values


def reconstruct_nonparam_kernel(post_path: Path, kmax: int) -> np.ndarray:
    """
    Reconstruct non-parametric kernel samples (mu) from posterior components.
    
    Model: non_parametric_monotonic_empirical_bayes.stan
    
    mu[k, nu] = inv_logit( logit_mu[k] )
    logit_mu[k] = logit( fmin(fmax(arg, 1e-12), 1-1e-12) )
    arg = p[k] * cumsum(Delta[k])
    p[k] = inv_logit(logit_p[k])
    """
    if not post_path.exists():
        raise FileNotFoundError(f"Nonparam posterior not found: {post_path}")

    # Read needed columns
    # We assume K=1 for this sweep
    df = pd.read_parquet(post_path)
    
    # 1. Get logit_p[1]
    msg = "logit_p[1]"
    if msg not in df.columns:
        raise ValueError(f"Missing {msg} in {post_path}")
    logit_p = df[msg].values
    p = 1.0 / (1.0 + np.exp(-logit_p))
    
    # 2. Get Delta[1, nu]
    # Delta columns should be Delta[1,1], Delta[1,2], ...
    # We need to find how many we have or matching kmax
    delta_cols = [c for c in df.columns if re.match(r"Delta\[1,\d+\]", c)]
    # Sort by index
    delta_cols.sort(key=lambda x: int(x.split(',')[-1].rstrip(']')))
    
    # If we have fewer columns than kmax, it means the model might have adapted to data size?
    # Or kmax in data was smaller.
    # Let's use the number of columns found as the effective kmax for reconstruction.
    n_deltas = len(delta_cols)
    if n_deltas == 0:
        raise ValueError(f"No Delta columns found in {post_path}")
        
    Delta = df[delta_cols].values # shape (n_samples, n_deltas)
    
    # Reconstruct
    # cumsum along axis 1
    cumsum_Delta = np.cumsum(Delta, axis=1)
    
    # arg = p * cumsum
    # p is (n_samples,), make it (n_samples, 1)
    arg = p[:, None] * cumsum_Delta
    
    # Clip to avoid numeric issues (matching Stan)
    arg = np.clip(arg, 1e-12, 1.0 - 1e-12)
    
    # logit_mu = logit(arg) -> mu = arg
    # Since mu = inv_logit(logit_mu), and logit_mu = logit(arg), mu is just arg!
    # (The Stan model has that identity transform at the end: mu = inv_logit(logit_mu))
    mu = arg
    
    return mu


def get_inference_diagnostics(results_dir: Path, model_subdir: str) -> dict:
    """Extract max R_hat and min ESS/bulk from inference summary."""
    sum_path = results_dir / model_subdir / "inference_summary.parquet"
    if not sum_path.exists():
        return {}
    
    df = pd.read_parquet(sum_path)
    # R_hat column
    rhat = df["R_hat"].max() if "R_hat" in df.columns else np.nan
    ess = df["ESS_bulk"].min() if "ESS_bulk" in df.columns else np.nan
    
    return {
        f"max_rhat_{model_subdir}": rhat,
        f"min_ess_{model_subdir}": ess,
    }

def process_dataset(data_name: str, 
                   synthetic_root: Path, 
                   results_root: Path) -> dict:
    
    # 1. Load Metadata
    data_dir = synthetic_root / data_name
    row_meta = load_metadata(data_dir)
    
    # 2. Paths
    res_dir = results_root / "synthetic" / data_name / "K~1"
    simple_subdir = "stan_file~model_simple_contagion"
    nonparam_subdir = "stan_file~non_parametric_monotonic_empirical_bayes"
    
    # Check if results exist
    if not (res_dir / simple_subdir).exists() or not (res_dir / nonparam_subdir).exists():
        logger.warning(f"Missing results for {data_name}")
        return None
    
    # 3. Load Posteriors
    try:
        rho_samples = load_simple_posterior(res_dir)
        # Assuming kmax from metadata for reconstruction, 
        # but let's just use what's in the posterior columns to be safe/dynamic
        mu_samples = reconstruct_nonparam_kernel(
            res_dir / nonparam_subdir / "posterior_samples.parquet", 
            kmax=row_meta["kmax"]
        )
    except Exception as e:
        logger.error(f"Error loading posteriors for {data_name}: {e}")
        return None

    # Align dimensions
    # mu_samples is (n_samples, k_effective)
    # rho_samples is (n_samples_simple,)
    # If distinct counts, take min
    n_min = min(len(rho_samples), len(mu_samples))
    rho_samples = rho_samples[:n_min]
    mu_samples = mu_samples[:n_min]
    
    # 4. Compute Metrics
    
    # A) L1 Complexity
    # Reference kernel from simple fit
    # We must match the k dimension of mu_samples
    k_eff = mu_samples.shape[1]
    
    # simple_contagion_reference expects scalar rho, returns vector
    # We vectorize? Or just compute array
    # simple kernel = 1 - (1-rho)^nu
    rho_hat = np.mean(rho_samples)
    mu_mean = np.mean(mu_samples, axis=0)

    simple_ref = simple_contagion_reference(rho_hat, k_eff)
    l1_mean = np.mean(np.abs(mu_mean - simple_ref))
    
    # B) Malahbois Complexity
    # This computes Mahalanobis distance of the posterior samples relative to the simple manifold
    # It internally computes the reference from the MEAN of rho_samples passed in
    try:
        mal_res = compute_malahbois_complexity(mu_samples, rho_samples)
        mahal_non_centrality = mal_res.non_centrality
        mahal_mode_non_centrality = mal_res.mode_non_centrality
        mahal_median_total = np.median(mal_res.total_distance)
        mahal_median_centered = np.median(mal_res.centered_distance)
    except Exception as e:
        logger.warning(f"Malahbois failed for {data_name}: {e}")
        mahal_non_centrality = np.nan
        mahal_mode_non_centrality = np.nan
        mahal_median_total = np.nan
        mahal_median_centered = np.nan

    # 5. Diagnostics
    diag_simple = get_inference_diagnostics(res_dir, simple_subdir)
    diag_nonparam = get_inference_diagnostics(res_dir, nonparam_subdir)
    
    # Combine all info
    return {
        **row_meta.to_dict(),
        "l1_mean": l1_mean,
        "mahal_non_centrality": mahal_non_centrality,
        "mahal_mode_non_centrality": mahal_mode_non_centrality,
        "mahal_median_total": mahal_median_total,
        "mahal_median_centered": mahal_median_centered,
        **diag_simple,
        **diag_nonparam,
    }


def main():
    parser = argparse.ArgumentParser(description="Aggregate sweep results into complexity table")
    parser.add_argument("--results-dir", type=Path, default="results/research_sim_sweep_sbm_daniel")
    parser.add_argument("--data-dir", type=Path, default="data/processed/synthetic")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    
    if args.output is None:
        args.output = args.results_dir / "complexity_table.parquet"
        
    print(f"Scanning results in {args.results_dir}")
    
    # Find all data_names in results
    # Structure: results/sweep/synthetic/data_name/...
    synthetic_res = args.results_dir / "synthetic"
    if not synthetic_res.exists():
        print(f"No synthetic results found at {synthetic_res}")
        return

    # Filter for sbm_daniel datasets
    data_names = [d.name for d in synthetic_res.iterdir() if d.is_dir() and d.name.startswith("sbm_daniel")]
    print(f"Found {len(data_names)} potential datasets")
    
    results = []
    for data_name in tqdm(data_names):
        try:
            res = process_dataset(data_name, args.data_dir, args.results_dir)
            if res:
                results.append(res)
        except Exception as e:
            logger.error(f"Failed to process {data_name}: {e}")
            
    if not results:
        print("No results aggregated.")
        return
        
    df = pd.DataFrame(results)
    print(f"\nAggregated {len(df)} rows.")
    print(df[["epsilon", "omega", "l1_mean", "mahal_non_centrality", "mahal_mode_non_centrality"]].head())
    
    df.to_parquet(args.output, index=False)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
