import pandas as pd
import numpy as np
import cmdstanpy
import argparse
import yaml
import json
import re
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


def _default_output_config():
    return {
        "posterior_mode": "full",  # full | compact | minimal
        "posterior_thin": None,
        "posterior_keep_regex": [
            r"^lp__$",
            r"^rho(\[|$)",
            r"^mu\[\d+,\d+\]$",
            r"^rho_l(\[|$)",
            r"^rho_h(\[|$)",
            r"^w(\[|$)",
            r"^k0(\[|$)",
        ],
        "posterior_float32": None,
        "parquet_compression": "zstd",
        "kernel_ci_level": 0.90,
        "kernel_map_bins": 128,
        "write_kernel_draws": False,
        "kernel_draws_thin": 1,
    }


def _resolve_output_config(config):
    output_cfg = dict(_default_output_config())
    output_cfg.update(config.get("output", {}))

    mode = str(output_cfg.get("posterior_mode", "full")).lower()
    if mode not in {"full", "compact", "minimal"}:
        raise ValueError("output.posterior_mode must be one of: full, compact, minimal")
    output_cfg["posterior_mode"] = mode

    if output_cfg["posterior_thin"] is None:
        output_cfg["posterior_thin"] = {"full": 1, "compact": 5, "minimal": 20}[mode]
    output_cfg["posterior_thin"] = max(1, int(output_cfg["posterior_thin"]))

    if output_cfg["posterior_float32"] is None:
        output_cfg["posterior_float32"] = mode != "full"
    output_cfg["posterior_float32"] = bool(output_cfg["posterior_float32"])

    output_cfg["kernel_draws_thin"] = max(1, int(output_cfg.get("kernel_draws_thin", 1)))
    output_cfg["kernel_ci_level"] = float(output_cfg.get("kernel_ci_level", 0.90))
    if not (0.0 < output_cfg["kernel_ci_level"] < 1.0):
        raise ValueError("output.kernel_ci_level must be in (0, 1).")
    output_cfg["kernel_map_bins"] = max(10, int(output_cfg.get("kernel_map_bins", 128)))
    return output_cfg


def _filter_posterior_draws(draws_df, output_cfg):
    mode = output_cfg["posterior_mode"]
    if mode == "full":
        selected = draws_df.copy()
    else:
        patterns = [re.compile(p) for p in output_cfg["posterior_keep_regex"]]
        keep_cols = [c for c in draws_df.columns if any(p.search(c) for p in patterns)]
        if not keep_cols and "lp__" in draws_df.columns:
            keep_cols = ["lp__"]
        selected = draws_df.loc[:, keep_cols].copy()

    thin = output_cfg["posterior_thin"]
    if thin > 1:
        selected = selected.iloc[::thin, :].reset_index(drop=True)

    if output_cfg["posterior_float32"]:
        for col in selected.columns:
            if pd.api.types.is_float_dtype(selected[col]):
                selected[col] = selected[col].astype(np.float32)
    return selected


def _marginal_map(samples, bins):
    x = np.asarray(samples, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    if np.allclose(x, x[0]):
        return float(x[0])
    n_bins = min(int(bins), max(20, int(np.sqrt(x.size))))
    hist, edges = np.histogram(x, bins=n_bins)
    idx = int(np.argmax(hist))
    return float(0.5 * (edges[idx] + edges[idx + 1]))


def _extract_kernel_draws(fit, kmax):
    try:
        mu = np.asarray(fit.stan_variable("mu"), dtype=float)
        if mu.ndim == 2:
            mu = mu[:, None, :]
        if mu.ndim == 3:
            km = min(int(kmax), int(mu.shape[2]))
            # Only return mu if it has finite values (generated quantities may be all NaN)
            if np.any(np.isfinite(mu[:, :, :km])):
                return mu[:, :, :km], "mu"
    except Exception:
        pass

    try:
        rho = np.asarray(fit.stan_variable("rho"), dtype=float)
        if rho.ndim == 1:
            rho = rho[:, None]
        if rho.ndim == 2:
            nu = np.arange(1, int(kmax) + 1, dtype=float)[None, None, :]
            kernels = 1.0 - np.power((1.0 - rho[:, :, None]), nu)
            return kernels, "rho"
    except Exception:
        pass

    try:
        rho_l = np.asarray(fit.stan_variable("rho_l"), dtype=float)
        rho_h = np.asarray(fit.stan_variable("rho_h"), dtype=float)
        w = np.asarray(fit.stan_variable("w"), dtype=float)
        k0 = np.asarray(fit.stan_variable("k0"), dtype=float)
        params = [rho_l, rho_h, w, k0]
        params = [p[:, None] if p.ndim == 1 else p for p in params]
        if all(p.ndim == 2 for p in params):
            k_components = min(p.shape[1] for p in params)
            rho_l, rho_h, w, k0 = [p[:, :k_components] for p in params]
            nu = np.arange(1, int(kmax) + 1, dtype=float)[None, None, :]
            kernels = rho_l[:, :, None] + (rho_h - rho_l)[:, :, None] / (
                1.0 + np.exp(-w[:, :, None] * (nu - k0[:, :, None]))
            )
            return kernels, "sigmoid"
    except Exception:
        pass

    return None, None


def _build_kernel_output_tables(kernel_draws, source, output_cfg, lp=None):
    n_draws, k_components, kmax = kernel_draws.shape
    ci_half = (1.0 - output_cfg["kernel_ci_level"]) / 2.0
    ci_low = 100.0 * ci_half
    ci_high = 100.0 * (1.0 - ci_half)

    # Joint MAP: the kernel from the single draw with highest log-posterior
    joint_map_draw_idx = None
    if lp is not None:
        lp_arr = np.asarray(lp, dtype=float).ravel()
        finite_mask = np.isfinite(lp_arr)
        if finite_mask.any():
            joint_map_draw_idx = int(np.argmax(np.where(finite_mask, lp_arr, -np.inf)))

    summary_rows = []
    draws_rows = []
    keep_draws = bool(output_cfg.get("write_kernel_draws", False))
    draws_thin = output_cfg["kernel_draws_thin"]

    for k in range(k_components):
        for nu_idx in range(kmax):
            samples = kernel_draws[:, k, nu_idx]
            finite = samples[np.isfinite(samples)]

            row = {
                "component": int(k + 1),
                "nu": int(nu_idx + 1),
                "mean": float(np.nanmean(samples)),
                "median": float(np.nanmedian(samples)),
                "map_marginal": _marginal_map(samples, bins=output_cfg["kernel_map_bins"]),
                "ci_low": float(np.percentile(finite, ci_low)) if finite.size > 0 else np.nan,
                "ci_high": float(np.percentile(finite, ci_high)) if finite.size > 0 else np.nan,
                "n_draws": int(finite.size),
                "source_param": source,
            }

            # Joint MAP value from the best draw
            if joint_map_draw_idx is not None:
                val = kernel_draws[joint_map_draw_idx, k, nu_idx]
                row["map_joint"] = float(val) if np.isfinite(val) else np.nan
            else:
                row["map_joint"] = np.nan

            summary_rows.append(row)

            if keep_draws:
                for draw_idx in range(0, n_draws, draws_thin):
                    draws_rows.append(
                        {
                            "draw": int(draw_idx),
                            "component": int(k + 1),
                            "nu": int(nu_idx + 1),
                            "value": float(kernel_draws[draw_idx, k, nu_idx]),
                            "source_param": source,
                        }
                    )

    summary_df = pd.DataFrame(summary_rows)
    draws_df = pd.DataFrame(draws_rows) if keep_draws else None
    return summary_df, draws_df


def run_stan_inference(summary_path, output_dir, instance_params, config_path=None):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if instance_params:
        if isinstance(instance_params, str):
            try:
                instance_params = json.loads(instance_params)
            except json.JSONDecodeError:
                # Fallback for old Snakemake string representation if needed
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
        # Dynamically update kmax to match the pre-computed matrices
        if M.shape[1] != kmax:
            print(f"Adjusting kmax from {kmax} to {M.shape[1]} to match input matrices.")
            kmax = M.shape[1]

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

    output_cfg = _resolve_output_config(config)
    parquet_compression = output_cfg.get("parquet_compression", "zstd")

    fit.summary().to_parquet(output_dir / "inference_summary.parquet", compression=parquet_compression)

    kernel_draws, kernel_source = _extract_kernel_draws(fit, kmax=kmax)
    if kernel_draws is not None:
        # Extract lp__ for joint MAP computation
        try:
            lp_values = np.asarray(fit.stan_variable("lp__"), dtype=float)
        except Exception:
            lp_values = None

        kernel_summary_df, kernel_draws_df = _build_kernel_output_tables(
            kernel_draws=kernel_draws,
            source=kernel_source,
            output_cfg=output_cfg,
            lp=lp_values,
        )
        kernel_summary_df.to_parquet(
            output_dir / "kernel_mode_summary.parquet",
            index=False,
            compression=parquet_compression,
        )
        if kernel_draws_df is not None and not kernel_draws_df.empty:
            kernel_draws_df.to_parquet(
                output_dir / "kernel_mode_draws.parquet",
                index=False,
                compression=parquet_compression,
            )

    posterior_df = fit.draws_pd()
    posterior_df = _filter_posterior_draws(posterior_df, output_cfg=output_cfg)
    posterior_df.to_parquet(
        output_dir / "posterior_samples.parquet",
        index=False,
        compression=parquet_compression,
    )
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--exposure-summary", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--instance-params", type=str, default="{}")
    args = parser.parse_args()
    
    run_stan_inference(args.exposure_summary, args.output_dir, args.instance_params, config_path=args.config)
