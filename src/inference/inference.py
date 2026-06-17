import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import cmdstanpy
import numpy as np
import pandas as pd
import yaml
from cmdstanpy import CmdStanMCMC


@dataclass
class StanFitResult:
    fit:            CmdStanMCMC
    stan_file_path: Path
    k_components:   int
    kmax:           int
    
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
        "empirical":              float(_logit(overall_p)),
        "empirical_floor":        float(_logit(floor_p)),
        "empirical_ceiling":      float(_logit(ceiling_p)),
        "empirical_rate":         overall_p,
        "empirical_floor_rate":   floor_p,
        "empirical_ceiling_rate": ceiling_p,
    }


# ---------------------------------------------------------------------------
# Stan data assembly
# ---------------------------------------------------------------------------

def _resolve_logit_param(value, empirical_logits, default):
    if value is None or (isinstance(value, str) and value in empirical_logits):
        return empirical_logits.get(value or "empirical", default)
    return float(value)


def _build_stan_data(stan_code, config_model, empirical_logits, K, M, N, kmax):
    stan_data = {"kmax": kmax, "V": M.shape[0], "K": K, "M": M, "N": N}

    # (stan_token, config_key, default, logit_resolved)
    TOKENS = [
        ("LOGIT_BETA_LOCATION",  "logit_beta_location",   -10.0, True),
        ("LOGIT_BETA_SCALE",     "logit_beta_scale",        2.0, False),
        ("SIGMA_SCALE",         "sigma_scale",              1.0, False),
        ("PI_CONCENTRATION",    "pi_concentration",         1.0, False),
        ("LOGIT_P_LOCATION",    "logit_p_location",         0.0, True),
        ("LOGIT_P_SCALE",       "logit_p_scale",            2.0, False),
        ("DELTA_CONCENTRATION", "delta_concentration",      1.0, False),
    ]
    for token, cfg_key, default, logit_resolved in TOKENS:
        if token in stan_code:
            value = config_model.get(cfg_key, default)
            stan_data[token] = (
                _resolve_logit_param(value, empirical_logits, default)
                if logit_resolved else float(value)
            )

    if "DEVIATION_SCALE" in stan_code:
        stan_data["DEVIATION_SCALE"] = [float(config_model.get("deviation_scale", 0.1))] * K

    return stan_data


# ---------------------------------------------------------------------------
# Main fitting function
# ---------------------------------------------------------------------------

def fit_stan(M, N, config_path, instance_params=None) -> StanFitResult:
    """Fit a Stan model and return a StanFitResult.

    M and N must already be sliced as appropriate
    (e.g. training rows only for CV); pass the full matrices for a standard fit.

    Parameters
    ----------
    M, N            : np.ndarray of shape (V, kmax)
    config_path     : str or Path — path to config.yaml
    instance_params : dict or JSON str — per-run overrides merged into config['model']
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

    M = np.asarray(M, dtype=float)
    N = np.asarray(N, dtype=float)
    kmax = M.shape[1]

    stan_file = config["model"]["stan_file"]
    if not stan_file.endswith(".stan"):
        stan_file += ".stan"
    stan_file_path = Path(stan_file)
    if not stan_file_path.exists():
        alt = Path("src/models") / stan_file
        if alt.exists():
            stan_file_path = alt
        else:
            raise FileNotFoundError(f"Stan file not found: {stan_file}")

    model        = cmdstanpy.CmdStanModel(stan_file=stan_file_path)
    stan_code    = stan_file_path.read_text()
    k_components = int(config["model"].get("K", 1))

    prior_floor = float(config["model"].get("empirical_prior_floor", 1e-6))
    prior_cap   = float(config["model"].get("empirical_prior_cap", 1.0 - 1e-6))
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
        f"Empirical logits: overall = {empirical_logits['empirical']:.4f}"
        f" (p = {empirical_logits['empirical_rate']:.6f}),"
        f" floor = {empirical_logits['empirical_floor']:.4f}"
        f" (p = {empirical_logits['empirical_floor_rate']:.6f}),"
        f" ceiling = {empirical_logits['empirical_ceiling']:.4f}"
        f" (p = {empirical_logits['empirical_ceiling_rate']:.6f})"
    )

    stan_data = _build_stan_data(
        stan_code, config["model"], empirical_logits, k_components, M, N, kmax)

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
    return StanFitResult(
        fit=fit,
        stan_file_path=stan_file_path,
        k_components=k_components,
        kmax=kmax,
    )

# ---------------------------------------------------------------------------
# Private kernel extraction functions
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


# ---------------------------------------------------------------------------
# Public summarization functions
# ---------------------------------------------------------------------------

def summarize_inference(result: StanFitResult) -> pd.DataFrame:
    return result.fit.summary()


def compute_model_comparison(result: StanFitResult) -> pd.DataFrame | None:
    """Compute WAIC and BIC from MCMC draws of pointwise log_lik.

    WAIC = -2 * (lppd - p_waic)
      lppd = sum_i log(mean_s exp(log_lik[s,i]))
      p_waic = sum_i var_s(log_lik[s,i])

    BIC = -2 * max_ll + n_params * log(n_obs)
      max_ll = max over draws of sum_i log_lik[s,i]
      n_params = global parameters only (not per-node latents)
    """
    fit            = result.fit
    stan_file_path = result.stan_file_path
    K              = result.k_components
    kmax           = result.kmax

    try:
        ll = np.asarray(fit.stan_variable("log_lik"), dtype=float)  # (S, V)
    except Exception as e:
        print(f"  model_comparison: could not extract log_lik ({e})")
        return None

    S, V = ll.shape

    # WAIC
    ll_max   = ll.max(axis=0, keepdims=True)
    lppd_vec = np.log(np.exp(ll - ll_max).sum(axis=0)) + ll_max[0] - np.log(S)
    lppd     = float(lppd_vec.sum())
    p_waic   = float(np.var(ll, axis=0).sum())
    waic     = -2.0 * (lppd - p_waic)

    # BIC
    ll_sum         = ll.sum(axis=1)
    max_log_lik    = float(ll_sum.max())
    mean_log_lik   = float(ll_sum.mean())
    median_log_lik = float(np.median(ll_sum))

    stan_file_name = Path(stan_file_path).name
    if "simple" in stan_file_name:
        n_params = K + 1 + max(0, K - 1)                   # logit_rho[K] + sigma + pi[K-1]
    elif "complex" in stan_file_name:
        n_params = 4 * K + 1 + max(0, K - 1)               # logit_rho_l[K], logit_rho_h[K], w[K], k0[K] + sigma + pi[K-1]
    elif "nonparametric" in stan_file_name:
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


def extract_kernel_summary(
    result: StanFitResult,
    ci_level: float = 0.90,
    map_bins: int = 128,
) -> pd.DataFrame | None:
    """Summarize kernel posterior draws into a per-(component, nu) DataFrame.

    Returns None if the model has no mu generated quantity.
    """
    kernel_draws, source = _extract_kernel_draws(result.fit, result.kmax)
    if kernel_draws is None:
        return None

    lp_values = np.asarray(result.fit.method_variables()["lp__"], dtype=float)
    n_draws, k_components, kmax = kernel_draws.shape

    ci_half = (1.0 - ci_level) / 2.0
    ci_low  = 100.0 * ci_half
    ci_high = 100.0 * (1.0 - ci_half)

    joint_map_draw_idx = None
    lp_arr = _align_lp_to_kernel_draws(lp_values, n_draws=n_draws)
    if lp_arr is not None:
        finite_mask = np.isfinite(lp_arr)
        if finite_mask.any():
            joint_map_draw_idx = int(np.argmax(np.where(finite_mask, lp_arr, -np.inf)))

    rows = []
    for k in range(k_components):
        for nu_idx in range(kmax):
            samples = kernel_draws[:, k, nu_idx]
            finite  = samples[np.isfinite(samples)]
            rows.append({
                "component":    int(k + 1),
                "nu":           int(nu_idx + 1),
                "mean":         float(np.nanmean(samples)),
                "median":       float(np.nanmedian(samples)),
                "map_marginal": _marginal_map(samples, bins=map_bins),
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
            })

    return pd.DataFrame(rows)


def get_posterior_draws(
    result: StanFitResult,
    thin: int = 1,
    float32: bool = False,
    keep_vars: list[str] | None = None,
) -> pd.DataFrame:
    """Return posterior draws as a DataFrame.

    Parameters
    ----------
    thin      : keep every nth row (1 = keep all)
    float32   : downcast float columns to float32
    keep_vars : list of regex strings to select columns; None keeps all
    """
    draws = result.fit.draws_pd()

    if keep_vars is not None:
        patterns  = [re.compile(p) for p in keep_vars]
        keep_cols = [c for c in draws.columns if any(p.search(c) for p in patterns)]
        if not keep_cols and "lp__" in draws.columns:
            keep_cols = ["lp__"]
        draws = draws.loc[:, keep_cols].copy()

    if thin > 1:
        draws = draws.iloc[::thin, :].reset_index(drop=True)

    if float32:
        for col in draws.columns:
            if pd.api.types.is_float_dtype(draws[col]):
                draws[col] = draws[col].astype(np.float32)

    return draws