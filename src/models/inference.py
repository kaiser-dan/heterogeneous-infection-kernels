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
from scipy.special import logsumexp as _logsumexp

from src.data_processing.operation_outbreak import generate_stan_matrices_oo
from src.data_processing.process_twitter import (
    generate_stan_matrices_twitter,
    _load_twitter_summary_df,
    _load_precomputed_twitter_mn,
)
from src.data_processing.process_twitter import load_twitter_MN
from src.data_processing.operation_outbreak import load_oo_MN

DATASET_LOADERS = {"twitter": load_twitter_MN, "oo": load_oo_MN, "synthetic": load_oo_MN}


def _to_rate_by_exposure(arr, row_weights=None):
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        if row_weights is None:
            return arr.sum(axis=0)
        w = np.asarray(row_weights, dtype=float).reshape(-1)
        if w.shape[0] != arr.shape[0]:
            raise ValueError(
                f"row_weights length ({w.shape[0]}) must match row count ({arr.shape[0]})."
            )
        return (arr * w[:, None]).sum(axis=0)
    raise ValueError(f"Unsupported array rank for exposure counts: {arr.ndim}")


def _logit(p):
    p = np.asarray(p, dtype=float)
    return np.log(p / (1.0 - p))


def _compute_empirical_logit_locations(
    M,
    N,
    prior_floor,
    prior_cap,
    floor_levels=3,
    ceiling_levels=4,
    row_weights=None,
):
    m_by_k = _to_rate_by_exposure(M, row_weights=row_weights)
    n_by_k = _to_rate_by_exposure(N, row_weights=row_weights)
    total = m_by_k + n_by_k

    def _clamp(p):
        return max(min(float(p), prior_cap), prior_floor)

    total_m = float(np.sum(m_by_k))
    total_n = float(np.sum(n_by_k))
    overall_p = _clamp(total_m / (total_m + total_n)) if (total_m + total_n) > 0 else prior_floor

    valid = total > 0
    if np.any(valid):
        rate = np.divide(m_by_k, total, out=np.full_like(m_by_k, np.nan), where=valid)
    else:
        rate = np.full_like(m_by_k, np.nan)

    low_slice = rate[:max(1, int(floor_levels))]
    low_slice = low_slice[np.isfinite(low_slice)]
    floor_p = _clamp(np.mean(low_slice)) if low_slice.size > 0 else overall_p

    high_slice = rate[-max(1, int(ceiling_levels)):]
    high_slice = high_slice[np.isfinite(high_slice)]
    ceiling_p = _clamp(np.max(high_slice)) if high_slice.size > 0 else overall_p

    return {
        "empirical": float(_logit(overall_p)),
        "empirical_floor": float(_logit(floor_p)),
        "empirical_ceiling": float(_logit(ceiling_p)),
        "empirical_rate": overall_p,
        "empirical_floor_rate": floor_p,
        "empirical_ceiling_rate": ceiling_p,
    }


def _resolve_logit_location(raw_value, empirical_logits, fallback):
    if isinstance(raw_value, str):
        key = raw_value.strip().lower()
        if key in empirical_logits:
            return float(empirical_logits[key])
    if raw_value is None:
        return float(fallback)
    return float(raw_value)


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


def _derive_defaults_for_mode(mode: str) -> dict:
    if mode == "full":
        return {"posterior_thin": 1, "posterior_float32": False}
    if mode == "compact":
        return {"posterior_thin": 5, "posterior_float32": True}
    if mode == "minimal":
        return {"posterior_thin": 25, "posterior_float32": True}
    raise ValueError(f"Unknown posterior_mode '{mode}'. Expected full|compact|minimal.")


def _resolve_output_config(config):
    output_cfg = dict(_default_output_config())
    output_cfg.update(config.get("output", {}))

    mode = str(output_cfg.get("posterior_mode", "full")).lower()
    if mode not in {"full", "compact", "minimal"}:
        raise ValueError("output.posterior_mode must be one of: full, compact, minimal")
    output_cfg["posterior_mode"] = mode

    mode_defaults = _derive_defaults_for_mode(mode)
    for key, val in mode_defaults.items():
        if output_cfg.get(key) is None:
            output_cfg[key] = val

    output_cfg["posterior_thin"] = max(1, int(output_cfg["posterior_thin"]))
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
    # Cap user-requested bins at sqrt(N) (or 20, whichever is larger) to avoid
    # noisy histograms when N is small.
    n_bins = min(int(bins), max(20, int(np.sqrt(x.size))))
    hist, edges = np.histogram(x, bins=n_bins)
    idx = int(np.argmax(hist))
    return float(0.5 * (edges[idx] + edges[idx + 1]))


def _extract_kernel_draws(fit, kmax: int):
    """Return (kernel_draws, source) by introspecting the fit's exposed Stan variables."""
    stan_vars = set(fit.metadata.stan_vars) if hasattr(fit, "metadata") else set()

    if "mu" in stan_vars:
        mu = np.asarray(fit.stan_variable("mu"), dtype=float)
        if mu.ndim == 2:
            mu = mu[:, None, :]
        if mu.ndim == 3:
            km = min(int(kmax), int(mu.shape[2]))
            # Only return mu if it has finite values (generated quantities may be all NaN)
            if np.any(np.isfinite(mu[:, :, :km])):
                return mu[:, :, :km], "mu"

    if "rho" in stan_vars:
        rho = np.asarray(fit.stan_variable("rho"), dtype=float)
        if rho.ndim == 1:
            rho = rho[:, None]
        if rho.ndim == 2:
            nu = np.arange(1, int(kmax) + 1, dtype=float)[None, None, :]
            kernels = 1.0 - np.power((1.0 - rho[:, :, None]), nu)
            return kernels, "rho"

    if all(v in stan_vars for v in ("rho_l", "rho_h", "w", "k0")):
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

    return None, None


def _align_lp_to_kernel_draws(lp, n_draws):
    if lp is None:
        return None
    lp_arr = np.asarray(lp, dtype=float)
    if lp_arr.size == 0:
        return None

    # CmdStanPy method_variables()["lp__"] is typically shaped
    # (iter_sampling, chains). Stan variable draw order is chain-major,
    # which corresponds to Fortran-order flattening for this shape.
    if lp_arr.ndim == 2:
        flat = lp_arr.ravel(order="F")
    else:
        flat = lp_arr.reshape(-1)

    if flat.size == n_draws:
        return flat
    if flat.size > n_draws:
        # Be permissive to interface/version differences that may include
        # extra draws; keep the most recent n_draws.
        return flat[-n_draws:]
    return None


def _build_kernel_output_tables(kernel_draws, source, output_cfg, lp=None):
    n_draws, k_components, kmax = kernel_draws.shape
    ci_half = (1.0 - output_cfg["kernel_ci_level"]) / 2.0
    ci_low = 100.0 * ci_half
    ci_high = 100.0 * (1.0 - ci_half)

    # Joint MAP: the kernel from the single draw with highest log-posterior
    joint_map_draw_idx = None
    if lp is not None:
        lp_arr = _align_lp_to_kernel_draws(lp, n_draws=n_draws)
        if lp_arr is None:
            lp_arr = np.full(n_draws, np.nan, dtype=float)
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


def _compute_model_comparison(fit, stan_data, stan_file_path, K, kmax):
    """Compute WAIC and BIC from MCMC draws of pointwise log_lik.

    WAIC = -2 * (lppd - p_waic)
      lppd = sum_i log(mean_s exp(log_lik[s,i]))
      p_waic = sum_i var_s(log_lik[s,i])

    BIC = -2 * max_ll + n_params * log(n_obs)
      max_ll = max over draws of sum_i log_lik[s,i]
      n_params = global parameters only (not per-node latents)
    """
    try:
        ll = np.asarray(fit.stan_variable("log_lik"), dtype=float)  # (S, V)
    except Exception as e:
        print(f"  model_comparison: could not extract log_lik ({e})")
        return None

    S, V = ll.shape

    # --- WAIC ---
    # lppd per observation: log mean_s exp(ll[s,i]) = logsumexp(ll[:,i]) - log(S)
    ll_max = ll.max(axis=0, keepdims=True)
    lppd_vec = np.log(np.exp(ll - ll_max).sum(axis=0)) + ll_max[0] - np.log(S)
    lppd = float(lppd_vec.sum())
    p_waic = float(np.var(ll, axis=0).sum())
    waic = -2.0 * (lppd - p_waic)

    # --- BIC ---
    ll_sum = ll.sum(axis=1)          # log-likelihood sum per draw, shape (S,)
    max_log_lik = float(ll_sum.max())
    mean_log_lik = float(ll_sum.mean())
    # Median is more robust than max as an MLE surrogate when MCMC mixes poorly
    # (e.g. label switching in K>1 models where max can be biased down by stuck chains).
    median_log_lik = float(np.median(ll_sum))

    stan_file_name = Path(stan_file_path).name
    n_obs = V
    if "simple_contagion" in stan_file_name:
        # logit_rho[K] + sigma + pi[K-1 free]
        n_params = K + 1 + max(0, K - 1)
    elif "complex_contagion" in stan_file_name:
        # logit_rho_l[K], logit_rho_h[K], w[K], k0[K] + sigma + pi[K-1]
        n_params = 4 * K + 1 + max(0, K - 1)
    elif "non_parametric" in stan_file_name:
        # logit_p[K] + Delta[K*(kmax-1)] + sigma + pi[K-1]
        n_params = K + K * (kmax - 1) + 1 + max(0, K - 1)
    else:
        n_params = None

    bic = (
        -2.0 * max_log_lik + n_params * np.log(n_obs)
        if n_params is not None else float("nan")
    )
    bic_median = (
        -2.0 * median_log_lik + n_params * np.log(n_obs)
        if n_params is not None else float("nan")
    )

    return pd.DataFrame([{
        "waic":           waic,
        "lppd":           lppd,
        "p_waic":         p_waic,
        "bic":            bic,
        "bic_median":     bic_median,
        "max_log_lik":    max_log_lik,
        "median_log_lik": median_log_lik,
        "mean_log_lik":   mean_log_lik,
        "n_obs":          n_obs,
        "n_params":       float(n_params) if n_params is not None else float("nan"),
        "n_samples":      S,
        "K":              K,
        "stan_file":      stan_file_name,
    }])


def compute_held_out_loglik(
    M_held: np.ndarray,
    N_held: np.ndarray,
    fit,
) -> dict:
    """Log posterior predictive density for held-out nodes.

    Uses population-level kernels mu[k] and mixture weights pi from the
    posterior -- no per-node latents required. Correct marginal predictive
    for a new node drawn from the population mixture.
    """
    eps = 1e-9

    if M_held.shape[0] == 0:
        return {"cv_score": 0.0, "n_held_out": 0, "mean_log_pred": np.nan}

    mu = np.asarray(fit.stan_variable("mu"), dtype=float)   # (S, K, kmax) or (S, kmax)
    pi = np.asarray(fit.stan_variable("pi"), dtype=float)   # (S, K) or (S,)

    # Normalise to (S, K, kmax) and (S, K) regardless of K=1 shape from CmdStanPy.
    if mu.ndim == 2:
        mu = mu[:, np.newaxis, :]   # (S, 1, kmax)
    if pi.ndim == 1:
        pi = pi[:, np.newaxis]      # (S, 1)

    S, K, kmax_model = mu.shape
    assert pi.shape == (S, K), (
        f"pi shape {pi.shape} does not match mu K-dim (S={S}, K={K}) after normalisation"
    )
    kmax_data = M_held.shape[1]
    k = min(kmax_model, kmax_data)

    mu_c = np.clip(mu[:, :, :k], eps, 1.0 - eps)    # (S, K, k)
    log_mu = np.log(mu_c)                             # (S, K, k)
    log_1m_mu = np.log(1.0 - mu_c)                   # (S, K, k)
    log_pi = np.log(np.clip(pi, eps, 1.0))            # (S, K)

    # Broadcast: (V, 1, 1, k) * (1, S, K, k) -> sum over k -> (V, S, K)
    M_b = M_held[:, np.newaxis, np.newaxis, :k]       # (V, 1, 1, k)
    N_b = N_held[:, np.newaxis, np.newaxis, :k]       # (V, 1, 1, k)
    log_mu_b = log_mu[np.newaxis, :, :, :]            # (1, S, K, k)
    log_1m_mu_b = log_1m_mu[np.newaxis, :, :, :]     # (1, S, K, k)

    ll_by_component = (M_b * log_mu_b + N_b * log_1m_mu_b).sum(axis=-1)  # (V, S, K)

    # Add log mixture weights and marginalise over K: (V, S)
    lps = ll_by_component + log_pi[np.newaxis, :, :]  # (V, S, K)
    log_lik = _logsumexp(lps, axis=-1)                 # (V, S)

    # Marginalise over posterior draws: (V,)
    log_pred = _logsumexp(log_lik, axis=-1) - np.log(S)  # (V,)

    cv_score = float(log_pred.sum())
    V_held = M_held.shape[0]
    return {
        "cv_score": cv_score,
        "n_held_out": V_held,
        "mean_log_pred": cv_score / V_held,
    }


def _cv_fold_mask(V: int, cv_folds: int, cv_fold: int) -> np.ndarray:
    if cv_folds < 1:
        raise ValueError(f"cv_folds must be >= 1, got {cv_folds}")
    if not (0 <= cv_fold < cv_folds):
        raise ValueError(f"cv_fold={cv_fold} out of range [0, {cv_folds})")
    return (np.arange(V) % cv_folds) == cv_fold


def run_stan_inference(summary_path, output_dir, instance_params, config_path=None):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if instance_params:
        # TODO: confirm dead before removing — all current callers (Snakemake, shell scripts)
        # pass JSON via json.dumps(), so json.loads() always succeeds; yaml.safe_load fallback
        # should never fire, but leave it until confirmed unused across all entry points.
        if isinstance(instance_params, str):
            try:
                instance_params = json.loads(instance_params)
            except json.JSONDecodeError:
                # Fallback for old Snakemake string representation if needed
                instance_params = yaml.safe_load(instance_params)
        
        config['model'].update(instance_params)
        for key in ['q', 'exposure_metric', 'source']:
            if key in instance_params: config[key] = instance_params[key]
        
    try:
        cmdstanpy.cmdstan_path()
    except ValueError as exc:
        raise RuntimeError(
            "CmdStan is not installed. Run `uv run python -m cmdstanpy.install_cmdstan` once."
        ) from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    source = config.get('source', 'twitter')
    kmax = config.get('kmax', 20)
    
    loader = DATASET_LOADERS.get(source)
    if loader is None:
        raise ValueError(f"Unknown source '{source}'. Expected one of {list(DATASET_LOADERS)}.")
    data = loader(summary_path, config, kmax)
    M, N = data["M"], data["N"]
    row_weights = data["row_weights"]
    kmax = data["kmax_observed"]
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
    
    stan_code = stan_file_path.read_text()
    
    from src.models.stan_registry import (
        ResolverContext,
        build_stan_data_extras,
        is_vector_deviation_scale,
        parse_data_block_names,
    )

    declared = parse_data_block_names(stan_code)

    # Check if model expects 2D (individual) or 1D (aggregated) matrices
    # Non-parametric models typically use array[V, kmax]
    requires_2d = bool(re.search(r"array\s*\[\s*V\s*,\s*kmax\s*\]", stan_code))
    
    if not requires_2d:
        print(f"Model {stan_file} appears to assume homogeneous population (1D matrices). Aggregating data...")
        # Aggregate across users (axis 0)
        M_agg = M.sum(axis=0)
        N_agg = N.sum(axis=0)

        # Override M and N with aggregated versions
        # Stan expects arrays, so keeping them as numpy arrays is fine (CmdStanPy handles it)
        M = M_agg
        N = N_agg
        row_weights = None
        # V is likely not in the data block for these models, but passing it usually doesn't hurt
        # unless strict data checking is enabled.

    # --- CV fold splitting ---
    cv_folds = int(config.get('cv_folds', 0))
    cv_fold  = int(config['model'].get('cv_fold', config.get('cv_fold', -1)))
    run_cv = cv_folds > 0 and cv_fold >= 0

    M_held = None
    N_held = None

    if run_cv:
        if not requires_2d:
            print(
                f"Warning: cv_folds={cv_folds} requested but model '{stan_file}' uses "
                "aggregated (1D) data — CV requires individual node rows. Skipping CV."
            )
            run_cv = False
        else:
            held_mask = _cv_fold_mask(V, cv_folds, cv_fold)
            train_mask = ~held_mask
            M_held = M[held_mask]
            N_held = N[held_mask]
            M = M[train_mask]
            N = N[train_mask]
            # row_weights is non-None for both OO (ones array) and Twitter (IPW);
            # None only occurs after 1D aggregation, which sets run_cv=False above.
            if row_weights is not None:
                row_weights = row_weights[train_mask]
            V = int(M.shape[0])
            print(
                f"CV fold {cv_fold}/{cv_folds}: "
                f"train={V} nodes, held_out={int(held_mask.sum())} nodes"
            )


    # Keep empirical Bayes configurable; a hard 0.01 floor is too aggressive
    # for sparse datasets where baseline adoption can be << 1%.
    prior_floor = float(config['model'].get('empirical_prior_floor', 1e-6))
    prior_cap = float(config['model'].get('empirical_prior_cap', 1.0 - 1e-6))
    if not (0.0 < prior_floor < prior_cap < 1.0):
        raise ValueError(
            f"Invalid empirical prior bounds: floor={prior_floor}, cap={prior_cap}. "
            "Need 0 < floor < cap < 1."
        )
    floor_levels = int(config['model'].get('empirical_floor_levels', 3))
    ceiling_levels = int(config['model'].get('empirical_ceiling_levels', 4))
    empirical_logits = _compute_empirical_logit_locations(
        M,
        N,
        prior_floor=prior_floor,
        prior_cap=prior_cap,
        floor_levels=floor_levels,
        ceiling_levels=ceiling_levels,
        row_weights=row_weights,
    )
    print(
        "Empirical logits:"
        f" overall={empirical_logits['empirical']:.4f} (p={empirical_logits['empirical_rate']:.6f}),"
        f" floor={empirical_logits['empirical_floor']:.4f} (p={empirical_logits['empirical_floor_rate']:.6f}),"
        f" ceiling={empirical_logits['empirical_ceiling']:.4f} (p={empirical_logits['empirical_ceiling_rate']:.6f})"
    )

    ctx = ResolverContext(
        empirical_logits=empirical_logits,
        fallback=0.0,
        kmax=kmax,
    )

    stan_data = {"kmax": kmax, "M": M, "N": N}

    if "ROW_WEIGHT" in declared:
        if row_weights is None:
            row_weights = np.ones(V, dtype=float)
        if len(row_weights) != V:
            raise ValueError(f"ROW_WEIGHT length ({len(row_weights)}) must match V ({V}).")
        stan_data["ROW_WEIGHT"] = np.asarray(row_weights, dtype=float)

    if "V" in declared:
        stan_data["V"] = V
    if "K" in declared and "K" in config["model"]:
        stan_data["K"] = k_components

    stan_data.update(build_stan_data_extras(stan_code, config["model"], ctx))

    if "DEVIATION_SCALE" in declared:
        if is_vector_deviation_scale(stan_code):
            stan_data["DEVIATION_SCALE"] = [dev_scale_val] * k_components
        else:
            stan_data["DEVIATION_SCALE"] = dev_scale_val
    
    if config.get('wandb_enabled'):
        wandb.init(project="heterogeneous-infection-kernels", name=f"{config['name']}_{output_dir.name}", config=config)
    
    sampling_cfg = config.get('model', {}).get('sampling', {})
    chains = int(sampling_cfg.get('chains', 1))
    iter_warmup = int(sampling_cfg.get('iter_warmup', 500))
    iter_sampling = int(sampling_cfg.get('iter_sampling', 500))
    adapt_delta = float(sampling_cfg.get('adapt_delta', 0.8))
    max_treedepth = sampling_cfg.get('max_treedepth')

    sample_kwargs = {
        'data': stan_data,
        'iter_sampling': iter_sampling,
        'iter_warmup': iter_warmup,
        'chains': chains,
        'adapt_delta': adapt_delta,
    }
    if max_treedepth is not None:
        sample_kwargs['max_treedepth'] = int(max_treedepth)

    fit = model.sample(**sample_kwargs)

    output_cfg = _resolve_output_config(config)
    parquet_compression = output_cfg.get("parquet_compression", "zstd")

    fit.summary().to_parquet(output_dir / "inference_summary.parquet", compression=parquet_compression)

    mc_df = _compute_model_comparison(fit, stan_data, stan_file, k_components, kmax)
    if mc_df is not None:
        mc_df.to_parquet(output_dir / "model_comparison.parquet", index=False, compression=parquet_compression)
        print(f"  WAIC={mc_df['waic'].iloc[0]:.2f}  BIC={mc_df['bic'].iloc[0]:.2f}  n_params={mc_df['n_params'].iloc[0]:.0f}")

    kernel_draws, kernel_source = _extract_kernel_draws(fit, kmax=kmax)
    if kernel_draws is not None:
        # Extract lp__ for joint MAP computation
        lp_values = np.asarray(fit.method_variables()["lp__"], dtype=float)

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

    if run_cv and M_held is not None and M_held.shape[0] > 0:
        cv_result = compute_held_out_loglik(M_held, N_held, fit)
        cv_df = pd.DataFrame([{
            "cv_score":       cv_result["cv_score"],
            "n_held_out":     cv_result["n_held_out"],
            "mean_log_pred":  cv_result["mean_log_pred"],
            "cv_fold":        cv_fold,
            "cv_folds":       cv_folds,
        }])
        cv_df.to_parquet(
            output_dir / "cv_held_out_loglik.parquet",
            index=False,
            compression=parquet_compression,
        )
        print(
            f"CV fold {cv_fold}: cv_score={cv_result['cv_score']:.4f}  "
            f"mean_log_pred={cv_result['mean_log_pred']:.4f}"
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
