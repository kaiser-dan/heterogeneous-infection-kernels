import pandas as pd
import numpy as np
import cmdstanpy
import argparse
import yaml
import json
from pathlib import Path
import wandb
import os

def generate_stan_matrices_twitter(summary_df, q, metric, kmax):
    V = len(summary_df)
    M = np.zeros((V, kmax), dtype=float) # Change to float for weights
    N = np.zeros((V, kmax), dtype=float)
    
    # Get weights if they exist
    weights = summary_df.get('ipw', np.ones(V)).values
    
    for i, row in enumerate(summary_df.itertuples()):
        A = row.A_array
        w = weights[i]
        
        if q == 1.0:
            level = sum(A) if metric == 'k' else len(A)
        else:
            if metric == 'k':
                level = sum([np.random.binomial(a, q) for a in A])
            else:
                probs = [1.0 - (1.0 - q)**a for a in A]
                level = sum([np.random.binomial(1, p) for p in probs])
        if level <= 0: continue
        idx = min(level, kmax) - 1
        
        if row.adopted: 
            M[i, idx] = 1.0 * w
        else: 
            N[i, idx] = 1.0 * w
            
    return M, N

def generate_stan_matrices_oo(ir_path, config):
    """
    Discretize OO dynamic graph into Stan matrices.
    ir_path is a directory containing edges.parquet and infections.parquet
    """
    ir_path = Path(ir_path)
    
    # Check for pre-computed M/N matrices
    m_path = ir_path / "M.parquet"
    n_path = ir_path / "N.parquet"
    
    if m_path.exists() and n_path.exists():
        print(f"Loading pre-computed OO matrices from {ir_path}")
        M_df = pd.read_parquet(m_path)
        N_df = pd.read_parquet(n_path)
        # Convert to numpy and ensure float type for Stan real arrays
        return M_df.values.astype(float), N_df.values.astype(float)

    print(f"Warning: Pre-computed OO matrices not found in {ir_path}. Falling back to dynamic discretization (not yet fully implemented).")
    edges = pd.read_parquet(ir_path / "edges.parquet")
    infections = pd.read_parquet(ir_path / "infections.parquet")
    
    kmax = config.get('kmax', 20)
    # Mock return for fallback
    V_mock = len(infections['user_id'].unique()) if 'user_id' in infections.columns else 100
    M = np.zeros((V_mock, kmax), dtype=float)
    N = np.zeros((V_mock, kmax), dtype=float)
    return M, N

def run_stan_inference(summary_path, output_dir, instance_params, config_path=None):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if instance_params:
        if isinstance(instance_params, str):
            # Use yaml.safe_load as it is more robust than json.loads for Snakemake's string representation
            instance_params = yaml.safe_load(instance_params)
        config['model'].update(instance_params)
        for key in ['q', 'exposure_metric', 'source']:
            if key in instance_params: config[key] = instance_params[key]
        
    print(f"Debug: Exposure Summary Path: {summary_path}")
    if os.path.exists(summary_path):
        print(f"Debug: Exists. Is Dir: {os.path.isdir(summary_path)}")
        if os.path.isdir(summary_path):
             print(f"Debug: Contents: {os.listdir(summary_path)}")
    else:
        print(f"Debug: Does NOT exist!")

    # Check CmdStan
    try:
        from cmdstanpy import cmdstan_path, install_cmdstan
        print(f"CmdStanPy CmdStan path: {cmdstan_path()}")
    except Exception as e:
        print(f"CmdStan not found or error: {e}")
        print("Installing CmdStan...")
        from cmdstanpy import install_cmdstan
        install_cmdstan()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    source = config.get('source', 'twitter')
    # ... rest of function
    kmax = config.get('kmax', 20)
    
    if source == 'twitter':
        summary_df = pd.read_parquet(summary_path)
        V_max_local = config.get('model', {}).get('V_max_local', 200)
        
        # Stratified Sampling with Inverse Probability Weighting (IPW)
        adopters = summary_df[summary_df['adopted']]
        non_adopters = summary_df[~summary_df['adopted']]
        
        n_adopters = len(adopters)
        n_non_adopters = len(non_adopters)
        
        # Goal: preserve ALL adopters if possible, fill rest with non-adopters
        # If too many adopters, we sample them too (and weight them)
        if len(summary_df) > V_max_local:
            # Determine quotas
            if n_adopters <= V_max_local // 2:
                # Take all adopters
                sample_a = adopters
                n_sample_a = n_adopters
                n_sample_n = V_max_local - n_sample_a
            else:
                # Too many adopters (rare case for this data), cap at 50%
                n_sample_a = V_max_local // 2
                sample_a = adopters.sample(n=n_sample_a, random_state=42)
                n_sample_n = V_max_local - n_sample_a
                
            # Sample non-adopters
            sample_n = non_adopters.sample(n=n_sample_n, random_state=42)
            
            # Calculate Weights (Inverse Probability)
            # W = Total / Sampled
            weight_a = n_adopters / n_sample_a if n_sample_a > 0 else 1.0
            weight_n = n_non_adopters / n_sample_n if n_sample_n > 0 else 1.0
            
            # Assign weights
            sample_a = sample_a.copy()
            sample_a['ipw'] = weight_a
            
            sample_n = sample_n.copy()
            sample_n['ipw'] = weight_n
            
            # Combine
            summary_df = pd.concat([sample_a, sample_n])
        else:
            summary_df['ipw'] = 1.0 # No subsampling
            
        q = config.get('q', 1.0)
        metric = config.get('exposure_metric', 'kappa')
        M, N = generate_stan_matrices_twitter(summary_df, q, metric, kmax)
         
        V = len(summary_df)
    else:
        # OO handles discretization via dynamic graph IR directory
        # summary_path for OO points to the 'effective_oo_...' directory
        M, N = generate_stan_matrices_oo(summary_path, config)
        V = M.shape[0]

    # Compile and load model to check data requirements
    stan_file = config['model']['stan_file']
    if not stan_file.endswith(".stan"):
        stan_file += ".stan"
    
    stan_file_path = Path(stan_file)
    if not stan_file_path.exists():
        # Check in src/models
        alt_path = Path("src/models") / stan_file
        if alt_path.exists():
            stan_file_path = alt_path
        else:
            raise FileNotFoundError(f"Could not find Stan file: {stan_file} in . or src/models/")
            
    model = cmdstanpy.CmdStanModel(stan_file=stan_file_path)
    
    # Extract model parameters
    k_components = int(config['model'].get('K', 1))
    dev_scale_val = float(config['model'].get('deviation_scale', 0.1))
    
    # Check model code to see if it expects a vector or scalar for DEVIATION_SCALE
    # A simple but effective way is to look at the .stan file content
    # Check model code to see if it expects a vector or scalar for DEVIATION_SCALE
    # A simple but effective way is to look at the .stan file content
    stan_code = stan_file_path.read_text()
    
    # Use regex to be more robust than simple string matching
    # We want to match "vector[...] DEVIATION_SCALE" or similar
    # but NOT "real DEVIATION_SCALE"
    import re
    # Match "vector" followed eventually by "DEVIATION_SCALE", allowing for <lower=...> etc
    # The key is that "vector" should be the type keyword
    # This regex allows for:
    # vector DEVIATION_SCALE
    # vector[K] DEVIATION_SCALE
    # vector<lower=0>[K] DEVIATION_SCALE
    vector_decl_pattern = re.compile(r"vector\b.*DEVIATION_SCALE")
    is_vector_dev = bool(vector_decl_pattern.search(stan_code))
    
    is_jgy_mod = "REPLICATES" in stan_code
    
    # Check if model expects 2D (individual) or 1D (aggregated) matrices
    # Non-parametric models typically use array[V, kmax]
    requires_2d = "array[V, kmax]" in stan_code or "array[V,kmax]" in stan_code
    
    if not requires_2d:
        print(f"Model {stan_file} appears to assume homogeneous population (1D matrices). Aggregating data...")
        # Aggregate across users (axis 0)
        M_agg = M.sum(axis=0)
        N_agg = N.sum(axis=0)
        
        # Override M and N with aggregated versions
        # Stan expects arrays, so keeping them as numpy arrays is fine (CmdStanPy handles it)
        M = M_agg
        N = N_agg
        # V is likely not in the data block for these models, but passing it usually doesn't hurt
        # unless strict data checking is enabled.
    
    logit_p_loc = config['model'].get('logit_p_location', 0.0)
    if logit_p_loc == 'empirical':
        total_m = M.sum()
        total_n = N.sum()
        emp_p = total_m / (total_m + total_n) if (total_m + total_n) > 0 else 0.01
        emp_p = max(min(emp_p, 0.99), 0.01)
        logit_p_loc = np.log(emp_p / (1 - emp_p))
        print(f"Empirical Bayes: Setting LOGIT_P_LOCATION to {logit_p_loc:.4f} (p={emp_p:.4f})")
    
    stan_data = {
        'kmax': kmax,
        'M': M,
        'N': N,
    }
    
    # Only add V and K if the model likely needs them (simple models might not)
    # But usually extra methods don't cause failure in generic CmdStan unless strict=True
    # safely adding them is better
    if re.search(r"\bV\b", stan_code) and "int" in stan_code:
        stan_data['V'] = V
    if re.search(r"\bK\b", stan_code) and "int" in stan_code and 'K' in config['model']:
        stan_data['K'] = k_components
         
    # Add other params robustly based on what's in the stan code
    if "CONCENTRATION" in stan_code:
        stan_data['CONCENTRATION'] = float(config['model'].get('concentration', 1.0))
        
    if "LOGIT_P_LOCATION" in stan_code:
        stan_data['LOGIT_P_LOCATION'] = float(logit_p_loc)
        
    if "LOGIT_P_SCALE" in stan_code:
        stan_data['LOGIT_P_SCALE'] = float(config['model'].get('logit_p_scale', 2.0))
    
    if is_vector_dev:
        stan_data['DEVIATION_SCALE'] = [dev_scale_val] * k_components
    elif "DEVIATION_SCALE" in stan_code:
        stan_data['DEVIATION_SCALE'] = dev_scale_val
        
    if is_jgy_mod:
        stan_data['REPLICATES'] = int(config['model'].get('replicates', 100))
    
    if config.get('wandb_enabled'):
        wandb.init(project="heterogeneous-infection-kernels", name=f"{config['name']}_{output_dir.name}", config=config)
    
    fit = model.sample(data=stan_data, iter_sampling=500, iter_warmup=500, chains=1)
    
    fit.summary().to_parquet(output_dir / "inference_summary.parquet")
    fit.draws_pd().to_parquet(output_dir / "posterior_samples.parquet")
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--exposure-summary", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--instance-params", type=str, default="{}")
    args = parser.parse_args()
    
    run_stan_inference(args.exposure_summary, args.output_dir, args.instance_params, config_path=args.config)
