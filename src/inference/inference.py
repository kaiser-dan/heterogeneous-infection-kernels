import argparse
import json
import re
from pathlib import Path

import cmdstanpy
import numpy as np
import pandas as pd
import yaml
from scipy.special import logsumexp as _logsumexp

from src.inference.stan_tokens import ResolverContext, build_stan_data_extras, parse_data_block_names


# ---------------------------------------------------------------------------
# Empirical logit helpers
# ---------------------------------------------------------------------------

def _logit(p):
    """Logit transform: log(p / (1 - p))."""
    p = np.asarray(p, dtype=float)
    return np.log(p / (1.0 - p))


def _compute_empirical_logit_locations(
    M,
    N,
    prior_floor,
    prior_cap,
    floor_levels=3,
    ceiling_levels=4,
):
    """Compute data-derived logit-scale prior locations for the EB model.

    Returns three logit-scale summaries: overall rate,
    mean of the lowest `floor_levels` exposure rates (prior floor), and max of
    the highest `ceiling_levels` rates (prior ceiling). All rates are clamped to
    [prior_floor, prior_cap] before the logit transform.
    """
    m_by_k = np.asarray(M, dtype=float).sum(axis=0)
    n_by_k = np.asarray(N, dtype=float).sum(axis=0)
    total = m_by_k + n_by_k

    def _clamp(p):
        return max(min(float(p), prior_cap), prior_floor)

    total_m = float(np.sum(m_by_k))
    total_n = float(np.sum(n_by_k))
    overall_p = _clamp(total_m / (total_m + total_n)) if (total_m + total_n) > 0 else prior_floor

    valid = total > 0
    rate = np.divide(m_by_k, total, out=np.full_like(m_by_k, np.nan), where=valid)

    low_slice = rate[:max(1, int(floor_levels))]
    low_slice = low_slice[np.isfinite(low_slice)]
    floor_p = _clamp(np.mean(low_slice)) if low_slice.size > 0 else overall_p

    high_slice = rate[-max(1, int(ceiling_levels)):]
    high_slice = high_slice[np.isfinite(high_slice)]
    ceiling_p = _clamp(np.max(high_slice)) if high_slice.size > 0 else overall_p

    return {
        "empirical":           float(_logit(overall_p)),
        "empirical_floor":     float(_logit(floor_p)),
        "empirical_ceiling":   float(_logit(ceiling_p)),
        "empirical_rate":      overall_p,
        "empirical_floor_rate":   floor_p,
        "empirical_ceiling_rate": ceiling_p,
    }


# ---------------------------------------------------------------------------
# Output config
# ---------------------------------------------------------------------------

def _default_output_config():
    """Return the baseline output config dict; overridden by config.yaml's output: section."""
    return {
        "posterior_mode":       "full",   # full | compact | minimal
        "posterior_thin":       None,
        "posterior_keep_regex": [
            r"^lp__$",
            r"^rho(\[|$)",
            r"^mu\[\d+,\d+\]$",
        ],
        "posterior_float32":    None,
        "parquet_compression":  "zstd",
        "kernel_ci_level":      0.90,
        "kernel_map_bins":      128,
        "write_kernel_draws":   False,
        "kernel_draws_thin":    1,
    }


def _derive_defaults_for_mode(mode: str) -> dict:
    """Map posterior_mode (full/compact/minimal) to thinning and dtype defaults."""
    if mode == "full":
        return {"posterior_thin": 1,  "posterior_float32": False}
    if mode == "compact":
        return {"posterior_thin": 5,  "posterior_float32": True}
    if mode == "minimal":
        return {"posterior_thin": 25, "posterior_float32": True}
    raise ValueError(f"Unknown posterior_mode '{mode}'. Expected full|compact|minimal.")


def _resolve_output_config(config):
    """Merge config.yaml output: section over defaults, validate, and fill mode-derived values."""
    output_cfg = dict(_default_output_config())
    output_cfg.update(config.get("output", {}))

    mode = str(output_cfg.get("posterior_mode", "full")).lower()
    if mode not in {"full", "compact", "minimal"}:
        raise ValueError("output.posterior_mode must be one of: full, compact, minimal")
    output_cfg["posterior_mode"] = mode

    for key, val in _derive_defaults_for_mode(mode).items():
        if output_cfg.get(key) is None:
            output_cfg[key] = val

    output_cfg["posterior_thin"]     = max(1, int(output_cfg["posterior_thin"]))
    output_cfg["posterior_float32"]  = bool(output_cfg["posterior_float32"])
    output_cfg["kernel_draws_thin"]  = max(1, int(output_cfg.get("kernel_draws_thin", 1)))
    output_cfg["kernel_ci_level"]    = float(output_cfg.get("kernel_ci_level", 0.90))
    if not (0.0 < output_cfg["kernel_ci_level"] < 1.0):
        raise ValueError("output.kernel_ci_level must be in (0, 1).")
    output_cfg["kernel_map_bins"] = max(10, int(output_cfg.get("kernel_map_bins", 128)))
    return output_cfg


def _filter_posterior_draws(draws_df, output_cfg):
    """Thin and/or column-filter the raw posterior draws DataFrame.

    In full mode, returns all columns and rows. In compact/minimal mode,
    keeps only columns matching posterior_keep_regex, then thins rows by
    posterior_thin. Optionally downcasts floats to float32.
    """
    if output_cfg["posterior_mode"] == "full":
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


# ---------------------------------------------------------------------------
# Kernel extraction — mu only
# ---------------------------------------------------------------------------

def _extract_kernel_draws(fit, kmax: int):
    """Return (kernel_draws, 'mu') from the fit's mu generated quantity."""
    stan_vars = set(fit.metadata.stan_vars) if hasattr(fit, "metadata") else set()
    if "mu" not in stan_vars:
        return None, None
    mu = np.asarray(fit.stan_variable("mu"), dtype=float)
    if mu.ndim == 2:
        mu = mu[:, None, :]      # (S, 1, kmax) for K=1
    if mu.ndim != 3:
        return None, None
    km = min(int(kmax), int(mu.shape[2]))
    if not np.any(np.isfinite(mu[:, :, :km])):
        return None, None
    return mu[:, :, :km], "mu"


def _marginal_map(samples, bins):
    """Estimate the marginal MAP via histogram mode of the sample distribution."""
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


def _align_lp_to_kernel_draws(lp, n_draws):
    """Flatten CmdStanPy's lp__ array (iter_sampling × chains) to match kernel draw order."""
    if lp is None:
        return None
    lp_arr = np.asarray(lp, dtype=float)
    if lp_arr.size == 0:
        return None
    flat = lp_arr.ravel(order="F") if lp_arr.ndim == 2 else lp_arr.reshape(-1)
    if flat.size == n_draws:
        return flat
    if flat.size > n_draws:
        return flat[-n_draws:]
    return None


def _build_kernel_output_tables(kernel_draws, source, output_cfg, lp=None):
    """Summarize kernel posterior draws into a per-(component, nu) summary DataFrame.

    For each (k, ν) cell computes mean, median, marginal MAP, CI bounds, and
    — if lp is provided — the joint MAP value from the single highest-lp draw.
    Optionally also returns a thinned draws DataFrame when write_kernel_draws=True.
    """
    n_draws, k_components, kmax = kernel_draws.shape
    ci_half = (1.0 - output_cfg["kernel_ci_level"]) / 2.0
    ci_low  = 100.0 * ci_half
    ci_high = 100.0 * (1.0 - ci_half)

    joint_map_draw_idx = None
    if lp is not None:
        lp_arr = _align_lp_to_kernel_draws(lp, n_draws=n_draws)
        if lp_arr is not None:
            finite_mask = np.isfinite(lp_arr)
            if finite_mask.any():
                joint_map_draw_idx = int(np.argmax(np.where(finite_mask, lp_arr, -np.inf)))

    keep_draws  = bool(output_cfg.get("write_kernel_draws", False))
    draws_thin  = output_cfg["kernel_draws_thin"]
    summary_rows, draws_rows = [], []

    for k in range(k_components):
        for nu_idx in range(kmax):
            samples = kernel_draws[:, k, nu_idx]
            finite  = samples[np.isfinite(samples)]
            row = {
                "component":    int(k + 1),
                "nu":           int(nu_idx + 1),
                "mean":         float(np.nanmean(samples)),
                "median":       float(np.nanmedian(samples)),
                "map_marginal": _marginal_map(samples, bins=output_cfg["kernel_map_bins"]),
                "ci_low":       float(np.percentile(finite, ci_low))  if finite.size > 0 else np.nan,
                "ci_high":      float(np.percentile(finite, ci_high)) if finite.size > 0 else np.nan,
                "n_draws":      int(finite.size),
                "source_param": source,
                "map_joint":    (
                    float(kernel_draws[joint_map_draw_idx, k, nu_idx])
                    if joint_map_draw_idx is not None
                    and np.isfinite(kernel_draws[joint_map_draw_idx, k, nu_idx])
                    else np.nan
                ),
            }
            summary_rows.append(row)

            if keep_draws:
                for draw_idx in range(0, n_draws, draws_thin):
                    draws_rows.append({
                        "draw":         int(draw_idx),
                        "component":    int(k + 1),
                        "nu":           int(nu_idx + 1),
                        "value":        float(kernel_draws[draw_idx, k, nu_idx]),
                        "source_param": source,
                    })

    summary_df = pd.DataFrame(summary_rows)
    draws_df   = pd.DataFrame(draws_rows) if keep_draws else None
    return summary_df, draws_df


# ---------------------------------------------------------------------------
# Model comparison (WAIC + BIC)
# ---------------------------------------------------------------------------

def _compute_model_comparison(fit, stan_file_path, K, kmax):
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

    # WAIC
    ll_max  = ll.max(axis=0, keepdims=True)
    lppd_vec = np.log(np.exp(ll - ll_max).sum(axis=0)) + ll_max[0] - np.log(S)
    lppd    = float(lppd_vec.sum())
    p_waic  = float(np.var(ll, axis=0).sum())
    waic    = -2.0 * (lppd - p_waic)

    # BIC
    ll_sum         = ll.sum(axis=1)
    max_log_lik    = float(ll_sum.max())
    mean_log_lik   = float(ll_sum.mean())
    median_log_lik = float(np.median(ll_sum))

    stan_file_name = Path(stan_file_path).name
    if "simple_contagion" in stan_file_name:
        n_params = K + 1 + max(0, K - 1)          # logit_rho[K] + sigma + pi[K-1]
    elif "non_parametric" in stan_file_name:
        n_params = K + K * (kmax - 1) + 1 + max(0, K - 1)  # logit_p + Delta + sigma + pi
    else:
        n_params = None

    bic = (
        -2.0 * max_log_lik + n_params * np.log(V)
        if n_params is not None else float("nan")
    )
    bic_median = (
        -2.0 * median_log_lik + n_params * np.log(V)
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
        "n_obs":          V,
        "n_params":       float(n_params) if n_params is not None else float("nan"),
        "n_samples":      S,
        "K":              K,
        "stan_file":      stan_file_name,
    }])


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def compute_held_out_loglik(M_held: np.ndarray, N_held: np.ndarray, fit) -> dict:
    """Log posterior predictive density for held-out nodes.

    Uses population-level mu[k] and mixture weights pi — no per-node latents
    required. Correct marginal predictive for a new node from the population.
    """
    eps = 1e-9
    if M_held.shape[0] == 0:
        return {"cv_score": 0.0, "n_held_out": 0, "mean_log_pred": np.nan}

    mu = np.asarray(fit.stan_variable("mu"), dtype=float)   # (S, K, kmax) or (S, kmax)
    pi = np.asarray(fit.stan_variable("pi"), dtype=float)   # (S, K) or (S,)
    
    # Normalize to (S, K, kmax) and (S, K) regardless of K=1 shape from CmdStanPy.
    if mu.ndim == 2:
        mu = mu[:, np.newaxis, :]  # (S, 1, kmax)
    if pi.ndim == 1:
        pi = pi[:, np.newaxis]     # (S, 1)

    S, K, kmax_model = mu.shape
    assert pi.shape == (S, K)
    k = min(kmax_model, M_held.shape[1])

    mu_c    = np.clip(mu[:, :, :k], eps, 1.0 - eps)
    log_mu  = np.log(mu_c)
    log_1m  = np.log(1.0 - mu_c)
    log_pi  = np.log(np.clip(pi, eps, 1.0))

    M_b      = M_held[:, np.newaxis, np.newaxis, :k]
    N_b      = N_held[:, np.newaxis, np.newaxis, :k]
    log_mu_b = log_mu[np.newaxis, :, :, :]
    log_1m_b = log_1m[np.newaxis, :, :, :]

    ll_by_component = (M_b * log_mu_b + N_b * log_1m_b).sum(axis=-1)  # (V, S, K)
    lps      = ll_by_component + log_pi[np.newaxis, :, :]              # (V, S, K)
    log_lik  = _logsumexp(lps, axis=-1)                                # (V, S)
    log_pred = _logsumexp(log_lik, axis=-1) - np.log(S)               # (V,)

    cv_score = float(log_pred.sum())
    return {
        "cv_score":      cv_score,
        "n_held_out":    M_held.shape[0],
        "mean_log_pred": cv_score / M_held.shape[0],
    }


def _cv_fold_mask(V: int, cv_folds: int, cv_fold: int) -> np.ndarray:
    """Boolean mask of length V selecting the held-out nodes for a given fold.

    Assigns nodes to folds by index modulo cv_folds, so folds are interleaved
    rather than contiguous — each fold gets a representative spread of nodes.
    """
    if cv_folds < 1:
        raise ValueError(f"cv_folds must be >= 1, got {cv_folds}")
    if not (0 <= cv_fold < cv_folds):
        raise ValueError(f"cv_fold={cv_fold} out of range [0, {cv_folds})")
    return (np.arange(V) % cv_folds) == cv_fold


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_stan_inference(M, N, output_dir, instance_params, config_path):
    """Run Stan inference on pre-built M/N matrices.

    Parameters
    ----------
    M, N        : np.ndarray of shape (V, kmax) — infection and non-infection counts
    output_dir  : str or Path — where to write parquet outputs
    instance_params : dict or JSON str — per-run overrides merged into config['model']
    config_path : str or Path — path to config.yaml
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if instance_params:
        if isinstance(instance_params, str):
            instance_params = json.loads(instance_params)
        config["model"].update(instance_params)

    try:
        cmdstanpy.cmdstan_path()
    except ValueError as exc:
        raise RuntimeError(
            "CmdStan is not installed. Run `install_cmdstan` once."
        ) from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    M = np.asarray(M, dtype=float)
    N = np.asarray(N, dtype=float)
    V, kmax = M.shape

    # Stan model
    stan_file = config["model"]["stan_file"]
    if not stan_file.endswith(".stan"):
        stan_file += ".stan"
    stan_file_path = Path(stan_file)
    if not stan_file_path.exists():
        alt = Path("src/inference") / stan_file
        if alt.exists():
            stan_file_path = alt
        else:
            raise FileNotFoundError(f"Stan file not found: {stan_file}")

    model        = cmdstanpy.CmdStanModel(stan_file=stan_file_path)
    stan_code    = stan_file_path.read_text()
    declared     = parse_data_block_names(stan_code)
    k_components = int(config["model"].get("K", 1))

    # CV fold splitting
    cv_folds = int(config.get("cv_folds", 0))
    cv_fold  = int(config["model"].get("cv_fold", config.get("cv_fold", -1)))
    run_cv   = cv_folds > 0 and cv_fold >= 0

    M_held = N_held = None
    if run_cv:
        held_mask = _cv_fold_mask(V, cv_folds, cv_fold)
        M_held, N_held = M[held_mask], N[held_mask]
        M, N = M[~held_mask], N[~held_mask]
        V = M.shape[0]
        print(f"CV fold {cv_fold}/{cv_folds}: train={V}, held={held_mask.sum()}")

    # Empirical Bayes prior locations
    prior_floor    = float(config["model"].get("empirical_prior_floor", 1e-6))
    prior_cap      = float(config["model"].get("empirical_prior_cap", 1.0 - 1e-6))
    if not (0.0 < prior_floor < prior_cap < 1.0):
        raise ValueError(f"Invalid empirical prior bounds: floor={prior_floor}, cap={prior_cap}")
    empirical_logits = _compute_empirical_logit_locations(
        M, N,
        prior_floor=prior_floor,
        prior_cap=prior_cap,
        floor_levels=int(config["model"].get("empirical_floor_levels", 3)),
        ceiling_levels=int(config["model"].get("empirical_ceiling_levels", 4)),
    )
    print(
        f"Empirical logits: overall={empirical_logits['empirical']:.4f}"
        f" (p={empirical_logits['empirical_rate']:.6f}),"
        f" floor={empirical_logits['empirical_floor']:.4f}"
        f" (p={empirical_logits['empirical_floor_rate']:.6f}),"
        f" ceiling={empirical_logits['empirical_ceiling']:.4f}"
        f" (p={empirical_logits['empirical_ceiling_rate']:.6f})"
    )

    ctx = ResolverContext(empirical_logits=empirical_logits, fallback=0.0, kmax=kmax)

    # Build Stan data dict
    stan_data = {"kmax": kmax, "V": V, "K": k_components, "M": M, "N": N}
    stan_data.update(build_stan_data_extras(stan_code, config["model"], ctx))

    # DEVIATION_SCALE is vector[K] in current models — always inject as list
    if "DEVIATION_SCALE" in declared:
        dev_scale = float(config["model"].get("deviation_scale", 0.1))
        stan_data["DEVIATION_SCALE"] = [dev_scale] * k_components

    # Sampling
    sampling_cfg  = config.get("model", {}).get("sampling", {})
    sample_kwargs = {
        "data":          stan_data,
        "iter_sampling": int(sampling_cfg.get("iter_sampling", 500)),
        "iter_warmup":   int(sampling_cfg.get("iter_warmup", 500)),
        "chains":        int(sampling_cfg.get("chains", 1)),
        "adapt_delta":   float(sampling_cfg.get("adapt_delta", 0.8)),
    }
    if "max_treedepth" in sampling_cfg:
        sample_kwargs["max_treedepth"] = int(sampling_cfg["max_treedepth"])

    fit = model.sample(**sample_kwargs)

    # Outputs
    output_cfg         = _resolve_output_config(config)
    parquet_compression = output_cfg.get("parquet_compression", "zstd")

    fit.summary().to_parquet(
        output_dir / "inference_summary.parquet",
        compression=parquet_compression,
    )

    mc_df = _compute_model_comparison(fit, stan_file_path, k_components, kmax)
    if mc_df is not None:
        mc_df.to_parquet(
            output_dir / "model_comparison.parquet",
            index=False,
            compression=parquet_compression,
        )
        print(
            f"  WAIC={mc_df['waic'].iloc[0]:.2f}"
            f"  BIC={mc_df['bic'].iloc[0]:.2f}"
            f"  n_params={mc_df['n_params'].iloc[0]:.0f}"
        )

    kernel_draws, kernel_source = _extract_kernel_draws(fit, kmax=kmax)
    if kernel_draws is not None:
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

    fit.draws_pd().pipe(_filter_posterior_draws, output_cfg=output_cfg).to_parquet(
        output_dir / "posterior_samples.parquet",
        index=False,
        compression=parquet_compression,
    )

    if run_cv and M_held is not None and M_held.shape[0] > 0:
        cv_result = compute_held_out_loglik(M_held, N_held, fit)
        pd.DataFrame([{
            "cv_score":      cv_result["cv_score"],
            "n_held_out":    cv_result["n_held_out"],
            "mean_log_pred": cv_result["mean_log_pred"],
            "cv_fold":       cv_fold,
            "cv_folds":      cv_folds,
        }]).to_parquet(
            output_dir / "cv_held_out_loglik.parquet",
            index=False,
            compression=parquet_compression,
        )
        print(
            f"CV fold {cv_fold}: cv_score={cv_result['cv_score']:.4f}"
            f"  mean_log_pred={cv_result['mean_log_pred']:.4f}"
        )

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",          type=str, required=True)
    parser.add_argument("--M",               type=str, required=True, help="Path to M .npy file")
    parser.add_argument("--N",               type=str, required=True, help="Path to N .npy file")
    parser.add_argument("--output-dir",      type=str, required=True)
    parser.add_argument("--instance-params", type=str, default="{}")
    args = parser.parse_args()

    run_stan_inference(
        M=np.load(args.M),
        N=np.load(args.N),
        output_dir=args.output_dir,
        instance_params=args.instance_params,
        config_path=args.config,
    )
