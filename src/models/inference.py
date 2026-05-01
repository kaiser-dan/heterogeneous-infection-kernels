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


def _safe_logit(p):
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
        "empirical": float(_safe_logit(overall_p)),
        "empirical_floor": float(_safe_logit(floor_p)),
        "empirical_ceiling": float(_safe_logit(ceiling_p)),
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


def _binomial_level_probs(total_exposures, q):
    n = max(0, int(total_exposures))
    probs = np.zeros(n + 1, dtype=float)
    if n == 0 or q <= 0.0:
        probs[0] = 1.0
        return probs
    if q >= 1.0:
        probs[n] = 1.0
        return probs

    # Stable forward recursion for Binomial(n, q) PMF.
    probs[0] = (1.0 - q) ** n
    ratio = q / (1.0 - q)
    for k in range(n):
        probs[k + 1] = probs[k] * ((n - k) / float(k + 1)) * ratio

    total = probs.sum()
    if total > 0:
        probs /= total
    else:
        probs[0] = 1.0
    return probs


def _kappa_level_probs(A, q):
    counts = np.asarray(A, dtype=float)
    counts = counts[counts > 0]
    n = len(counts)

    probs = np.array([1.0], dtype=float)
    if n == 0 or q <= 0.0:
        return probs
    if q >= 1.0:
        out = np.zeros(n + 1, dtype=float)
        out[n] = 1.0
        return out

    success_probs = 1.0 - np.power(1.0 - q, counts)
    for p in success_probs:
        probs = np.convolve(probs, np.array([1.0 - p, p], dtype=float))

    total = probs.sum()
    if total > 0:
        probs /= total
    return probs


def _level_probabilities_twitter(A, q, metric):
    if metric == "k":
        return _binomial_level_probs(np.sum(np.asarray(A, dtype=float)), q)
    if metric == "kappa":
        return _kappa_level_probs(A, q)
    raise ValueError(f"Unsupported metric '{metric}'. Expected 'k' or 'kappa'.")


def _mn_row_from_level_probs(level_probs, adopted, kmax, mn_mode):
    M_vals = np.zeros(kmax, dtype=float)
    N_vals = np.zeros(kmax, dtype=float)
    if kmax <= 0:
        return M_vals, N_vals

    probs = np.asarray(level_probs, dtype=float).reshape(-1)
    if probs.size == 0:
        return M_vals, N_vals

    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if total <= 0:
        return M_vals, N_vals
    probs /= total

    max_level = probs.size - 1
    if max_level <= 0:
        return M_vals, N_vals

    mn_mode = str(mn_mode).lower()
    if mn_mode == "incidence":
        up = min(max_level, kmax)
        if adopted:
            M_vals[:up] = probs[1 : up + 1]
        else:
            N_vals[:up] = probs[1 : up + 1]
        # Event mass beyond kmax is right-censored into the terminal non-event bin.
        if max_level > kmax:
            N_vals[kmax - 1] += float(np.sum(probs[kmax + 1 :]))
        M_vals = np.clip(M_vals, 0.0, None)
        N_vals = np.clip(N_vals, 0.0, None)
        return M_vals, N_vals

    if mn_mode != "survival":
        raise ValueError(f"mn_mode must be 'incidence' or 'survival'. Got: {mn_mode}")

    cdf = np.cumsum(probs)
    nu = np.arange(1, kmax + 1, dtype=int)

    ge_mask = nu <= max_level
    gt_mask = nu < max_level

    # P(L >= nu)
    ge_vals = np.zeros(kmax, dtype=float)
    ge_vals[ge_mask] = 1.0 - cdf[nu[ge_mask] - 1]

    # P(L > nu)
    gt_vals = np.zeros(kmax, dtype=float)
    gt_vals[gt_mask] = 1.0 - cdf[nu[gt_mask]]

    if adopted:
        up = min(max_level, kmax)
        M_vals[:up] = probs[1 : up + 1]
        N_vals = gt_vals
    else:
        N_vals = ge_vals
    M_vals = np.clip(M_vals, 0.0, None)
    N_vals = np.clip(N_vals, 0.0, None)
    return M_vals, N_vals


def convert_incidence_to_adopters_only_survival(M):
    """
    Convert an incidence-style event matrix into an adopters-only survival target.

    Rows without any event mass are dropped. For retained rows:
    - M is unchanged
    - N[nu] is the probability mass of adopting strictly after nu

    For deterministic one-hot incidence rows, this yields the usual
    "add 1 to every pre-adoption bin" construction.
    """
    M = np.asarray(M, dtype=float)
    if M.ndim != 2:
        raise ValueError(f"M must be a 2D matrix. Got shape {M.shape}.")

    adopted_mask = M.sum(axis=1) > 0
    M_adopters = M[adopted_mask].copy()
    N_survival = np.zeros_like(M_adopters)

    if M_adopters.shape[1] > 1:
        tail_mass = np.flip(np.cumsum(np.flip(M_adopters, axis=1), axis=1), axis=1)
        N_survival[:, :-1] = tail_mass[:, 1:]

    M_adopters = np.clip(M_adopters, 0.0, None)
    N_survival = np.clip(N_survival, 0.0, None)
    return M_adopters, N_survival, adopted_mask


def generate_stan_matrices_twitter(summary_df, q, metric, kmax, mn_mode="incidence"):
    V = len(summary_df)
    M = np.zeros((V, kmax), dtype=float)
    N = np.zeros((V, kmax), dtype=float)
    mn_mode = str(mn_mode).lower()
    if mn_mode not in {"incidence", "survival"}:
        raise ValueError(f"mn_mode must be 'incidence' or 'survival'. Got: {mn_mode}")

    # Row-level IPW (if present) should weight the row likelihood in Stan,
    # not the per-cell M/N counts. This prevents double-weighting artifacts.
    if 'ipw' in summary_df.columns:
        row_weights = summary_df['ipw'].to_numpy(dtype=float)
    else:
        row_weights = np.ones(V, dtype=float)

    for i, row in enumerate(summary_df.itertuples()):
        level_probs = _level_probabilities_twitter(row.A_array, q, metric)
        m_row, n_row = _mn_row_from_level_probs(
            level_probs=level_probs,
            adopted=bool(row.adopted),
            kmax=kmax,
            mn_mode=mn_mode,
        )
        M[i, :] = m_row
        N[i, :] = n_row

    return M, N, row_weights

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


def _sorted_exposure_cols(columns):
    cols = [c for c in columns if c != "user_id"]
    try:
        return sorted(cols, key=lambda c: int(c))
    except ValueError:
        return sorted(cols)


def _parse_q_from_mn_filename(path, kind, metric):
    pattern = re.compile(rf"^{kind}_{re.escape(metric)}_q(.+)\.parquet$")
    match = pattern.match(Path(path).name)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _find_q_matched_mn_file(directory, kind, metric, q, atol=1e-9):
    directory = Path(directory)
    if not directory.exists():
        return None

    target_q = float(q)
    matches = []
    for path in sorted(directory.glob(f"{kind}_{metric}_q*.parquet")):
        q_val = _parse_q_from_mn_filename(path, kind=kind, metric=metric)
        if q_val is None:
            continue
        if abs(q_val - target_q) <= atol:
            matches.append(path)

    if not matches:
        return None
    if len(matches) > 1:
        print(
            f"Warning: multiple {kind}_{metric} parquet files matched q={target_q} in {directory}. "
            f"Using {matches[0].name}."
        )
    return matches[0]


def _read_twitter_mn_parquet(path):
    df = pd.read_parquet(path).copy()
    if "user_id" not in df.columns:
        df.insert(0, "user_id", pd.Series(np.arange(len(df)), dtype=str))

    value_cols = _sorted_exposure_cols(df.columns)
    if not value_cols:
        raise ValueError(f"{path} has no exposure columns.")

    for col in value_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)

    df["user_id"] = df["user_id"].astype(str)
    df = df[["user_id"] + value_cols]
    # Be robust to accidental duplicates in upstream artifacts.
    df = df.groupby("user_id", as_index=False, sort=False)[value_cols].sum()
    return df


def _load_twitter_summary_df(summary_path):
    summary_path = Path(summary_path)
    if summary_path.is_dir():
        summary_file = summary_path / "summary.parquet"
        if not summary_file.exists():
            return None
        return pd.read_parquet(summary_file)
    return pd.read_parquet(summary_path)


def _aggregate_twitter_mn_files(paths):
    agg = None
    for path in paths:
        df = _read_twitter_mn_parquet(path).set_index("user_id")
        agg = df if agg is None else agg.add(df, fill_value=0.0)

    if agg is None:
        return None

    agg = agg.fillna(0.0)
    agg.columns = _sorted_exposure_cols(["user_id"] + list(agg.columns))
    return agg


def _resolve_twitter_processed_dir(config, summary_path):
    candidates = []
    twitter_cfg = config.get("twitter", {}) if isinstance(config.get("twitter", {}), dict) else {}

    # Highest-priority explicit overrides.
    for key in ("twitter_processed_dir", "processed_dir"):
        val = config.get(key)
        if val:
            candidates.append(Path(val))
    val = twitter_cfg.get("processed_dir")
    if val:
        candidates.append(Path(val))

    # Data-prep config referenced by the sweep config.
    data_config_path = config.get("data_config")
    if data_config_path:
        cfg_path = Path(data_config_path)
        if not cfg_path.is_absolute():
            cfg_path = Path.cwd() / cfg_path
        if cfg_path.exists():
            try:
                with open(cfg_path, "r") as f:
                    prep_cfg = yaml.safe_load(f) or {}
                prep_dir = prep_cfg.get("processed_dir")
                if prep_dir:
                    candidates.append(Path(prep_dir))
            except Exception as e:
                print(f"Warning: could not load data_config '{cfg_path}': {e}")

    # Conservative fallbacks.
    summary_name = Path(summary_path).stem.lower()
    if "survival" in summary_name:
        candidates.append(Path("data/processed/twitter_survival"))
    candidates.append(Path("data/processed/twitter"))

    for candidate in candidates:
        candidate = Path(candidate)
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        if candidate.exists():
            return candidate
    return None


def _load_precomputed_twitter_mn(summary_df, config, summary_path, q, metric):
    summary_df = summary_df.copy() if summary_df is not None else pd.DataFrame()
    if not summary_df.empty and "user_id" in summary_df.columns and summary_df["user_id"].astype(str).duplicated().any():
        # Without intervention_id in summary rows, duplicate user_id rows are ambiguous.
        print(
            "Warning: summary has duplicate user_id rows; cannot align pre-computed "
            "Twitter M/N uniquely. Falling back to A_array reconstruction."
        )
        return None

    summary_path = Path(summary_path)
    m_indexed = None
    n_indexed = None
    source_desc = None

    if summary_path.is_dir():
        direct_m = summary_path / "M.parquet"
        direct_n = summary_path / "N.parquet"
        if direct_m.exists() and direct_n.exists():
            m_indexed = _read_twitter_mn_parquet(direct_m).set_index("user_id")
            n_indexed = _read_twitter_mn_parquet(direct_n).set_index("user_id")
            source_desc = f"{summary_path} (direct)"

    if m_indexed is None or n_indexed is None:
        processed_dir = _resolve_twitter_processed_dir(config, summary_path)
        if processed_dir is None:
            return None

        full_dir = processed_dir / "full"
        m_full = _find_q_matched_mn_file(full_dir, kind="M", metric=metric, q=q)
        n_full = _find_q_matched_mn_file(full_dir, kind="N", metric=metric, q=q)
        if m_full is not None and n_full is not None:
            m_indexed = _read_twitter_mn_parquet(m_full).set_index("user_id")
            n_indexed = _read_twitter_mn_parquet(n_full).set_index("user_id")
            source_desc = f"{processed_dir}/full"
        else:
            m_files = []
            n_files = []
            for d in sorted(processed_dir.iterdir()):
                if not d.is_dir() or d.name in {"full", "summaries"}:
                    continue
                m_path = _find_q_matched_mn_file(d, kind="M", metric=metric, q=q)
                n_path = _find_q_matched_mn_file(d, kind="N", metric=metric, q=q)
                if m_path is not None and n_path is not None:
                    m_files.append(m_path)
                    n_files.append(n_path)

            if m_files and n_files:
                m_indexed = _aggregate_twitter_mn_files(m_files)
                n_indexed = _aggregate_twitter_mn_files(n_files)
                source_desc = f"{processed_dir}/* (aggregated)"

    if m_indexed is None or n_indexed is None:
        return None

    # Ensure aligned exposure columns.
    exposure_cols = _sorted_exposure_cols(["user_id"] + list(set(m_indexed.columns) | set(n_indexed.columns)))
    m_indexed = m_indexed.reindex(columns=exposure_cols, fill_value=0.0)
    n_indexed = n_indexed.reindex(columns=exposure_cols, fill_value=0.0)

    if summary_df.empty or "user_id" not in summary_df.columns:
        selected_ids = list(m_indexed.index)
    else:
        selected_ids = summary_df["user_id"].astype(str).tolist()
        missing = sorted(set(selected_ids) - set(m_indexed.index))
        if missing:
            # Summary and processed artifacts are out of sync; fall back to reconstruction.
            print(
                f"Warning: precomputed Twitter M/N missing {len(missing)} users "
                f"(example: {missing[:3]}). Falling back to summary reconstruction."
            )
            return None

    M = m_indexed.loc[selected_ids, exposure_cols].to_numpy(dtype=float)
    N = n_indexed.loc[selected_ids, exposure_cols].to_numpy(dtype=float)
    return M, N, source_desc


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
    
    row_weights = None

    if source == 'twitter':
        loaded_summary_df = _load_twitter_summary_df(summary_path)
        summary_df = loaded_summary_df.copy() if loaded_summary_df is not None else pd.DataFrame()
        if not summary_df.empty and "user_id" in summary_df.columns:
            summary_df["user_id"] = summary_df["user_id"].astype(str)
        V_max_local = config.get('model', {}).get('V_max_local', 200)
        twitter_cfg = config.get('twitter', {}) if isinstance(config.get('twitter', {}), dict) else {}
        adopters_only = bool(
            twitter_cfg.get(
                'adopters_only',
                config.get('twitter_adopters_only', config.get('model', {}).get('twitter_adopters_only', False)),
            )
        )
        disable_ipw = bool(
            twitter_cfg.get(
                'disable_ipw',
                config.get('twitter_disable_ipw', config.get('model', {}).get('twitter_disable_ipw', False)),
            )
        )

        if not summary_df.empty:
            if adopters_only:
                n_before = len(summary_df)
                summary_df = summary_df[summary_df['adopted']].copy()
                if summary_df.empty:
                    raise ValueError("Twitter adopters-only mode produced 0 rows.")
                # Case-only likelihood: unit row weights (no importance weighting).
                summary_df['ipw'] = 1.0
                print(f"Twitter mode: adopters-only ({len(summary_df)} / {n_before} rows).")
                print("Twitter mode: IPW disabled (unit ROW_WEIGHT).")
            else:
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
                    summary_df['ipw'] = 1.0  # No subsampling

                if disable_ipw:
                    summary_df = summary_df.copy()
                    summary_df['ipw'] = 1.0
                    print("Twitter mode: IPW explicitly disabled (unit ROW_WEIGHT).")
        else:
            print("Twitter mode: no summary.parquet found; relying on direct pre-computed M/N.")
        
        q = config.get('q', 1.0)
        metric = config.get('exposure_metric', 'kappa')
        twitter_mn_mode = str(
            config.get('twitter_mn_mode', config.get('model', {}).get('twitter_mn_mode', 'incidence'))
        ).lower()
        print(f"Twitter M/N mode: {twitter_mn_mode}")

        if not summary_df.empty and 'ipw' in summary_df.columns:
            row_weights = summary_df['ipw'].to_numpy(dtype=float)
        elif not summary_df.empty:
            row_weights = np.ones(len(summary_df), dtype=float)

        use_precomputed_twitter_mn = bool(
            config.get(
                'twitter_use_precomputed_mn',
                config.get('model', {}).get('twitter_use_precomputed_mn', True),
            )
        )

        loaded_precomputed = False
        if use_precomputed_twitter_mn:
            precomputed = _load_precomputed_twitter_mn(
                summary_df=summary_df,
                config=config,
                summary_path=summary_path,
                q=q,
                metric=metric,
            )
            if precomputed is not None:
                M, N, mn_source = precomputed
                loaded_precomputed = True
                if M.shape[1] != int(kmax):
                    print(
                        f"Adjusting kmax from {kmax} to {M.shape[1]} "
                        "to match pre-computed Twitter matrices."
                    )
                    kmax = int(M.shape[1])
                if row_weights is None:
                    row_weights = np.ones(M.shape[0], dtype=float)
                print(f"Twitter M/N source: pre-computed ({mn_source})")

        if not loaded_precomputed:
            if summary_df.empty:
                raise ValueError(
                    "Twitter inference requires summary.parquet or direct pre-computed M/N files."
                )
            M, N, row_weights = generate_stan_matrices_twitter(
                summary_df, q, metric, kmax, mn_mode=twitter_mn_mode
            )
            print("Twitter M/N source: reconstructed from summary A_array.")

        V = M.shape[0]
    else:
        # OO handles discretization via dynamic graph IR directory
        # summary_path for OO points to the 'effective_oo_...' directory
        M, N = generate_stan_matrices_oo(summary_path, config)

        oo_adopters_only = bool(
            config.get('oo', {}).get('adopters_only', False)
        )
        if oo_adopters_only:
            nodes_path = Path(summary_path) / "nodes.parquet"
            if not nodes_path.exists():
                raise FileNotFoundError(
                    f"oo.adopters_only=True but nodes.parquet not found at {nodes_path}"
                )
            nodes_df = pd.read_parquet(nodes_path)
            adopted_mask = nodes_df['adopted'].values.astype(bool)
            n_before = M.shape[0]
            M = M[adopted_mask]
            N = N[adopted_mask]
            print(f"OO adopters-only: kept {M.shape[0]} / {n_before} nodes (adopted=True).")

        drop_zero_bin = bool(
            config.get('model', {}).get(
                'drop_zero_exposure_bin',
                config.get('oo', {}).get('drop_zero_exposure_bin', False),
            )
        )
        if drop_zero_bin:
            if M.shape[1] <= 1:
                raise ValueError("Cannot drop zero-exposure bin: M/N have <=1 exposure column.")
            print("OO option enabled: dropping zero-exposure bin (column 0) before fitting.")
            M = M[:, 1:]
            N = N[:, 1:]
        V = M.shape[0]
        # Dynamically update kmax to match the pre-computed matrices
        if M.shape[1] != kmax:
            print(f"Adjusting kmax from {kmax} to {M.shape[1]} to match input matrices.")
            kmax = M.shape[1]
        row_weights = np.ones(V, dtype=float)

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
        row_weights = None
        # V is likely not in the data block for these models, but passing it usually doesn't hurt
        # unless strict data checking is enabled.
    
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

    logit_p_loc = _resolve_logit_location(
        config['model'].get('logit_p_location', 0.0),
        empirical_logits,
        fallback=0.0,
    )
    
    stan_data = {
        'kmax': kmax,
        'M': M,
        'N': N,
    }

    if "ROW_WEIGHT" in stan_code:
        if row_weights is None:
            row_weights = np.ones(V, dtype=float)
        if len(row_weights) != V:
            raise ValueError(f"ROW_WEIGHT length ({len(row_weights)}) must match V ({V}).")
        stan_data['ROW_WEIGHT'] = np.asarray(row_weights, dtype=float)
    
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

    if "SIGMA_SCALE" in stan_code:
        stan_data['SIGMA_SCALE'] = float(config['model'].get('sigma_scale', 1.0))
    if "PI_CONCENTRATION" in stan_code:
        stan_data['PI_CONCENTRATION'] = float(config['model'].get('pi_concentration', 1.0))
    if "LOGIT_MU_LOCATION" in stan_code:
        stan_data['LOGIT_MU_LOCATION'] = _resolve_logit_location(
            config['model'].get('logit_mu_location', 0.0),
            empirical_logits,
            fallback=0.0,
        )
    if "LOGIT_MU_SCALE" in stan_code:
        stan_data['LOGIT_MU_SCALE'] = float(config['model'].get('logit_mu_scale', 2.0))

    # SC model: logit-Normal prior on rho
    if "LOGIT_RHO_LOCATION" in stan_code:
        stan_data['LOGIT_RHO_LOCATION'] = _resolve_logit_location(
            config['model'].get('logit_rho_location', -10.0),
            empirical_logits,
            fallback=-10.0,
        )
    if "LOGIT_RHO_SCALE" in stan_code:
        stan_data['LOGIT_RHO_SCALE'] = float(config['model'].get('logit_rho_scale', 2.0))

    # CC model: logit-Normal priors on rho_l and rho_h
    if "LOGIT_RHO_L_LOCATION" in stan_code:
        stan_data['LOGIT_RHO_L_LOCATION'] = _resolve_logit_location(
            config['model'].get('logit_rho_l_location', -10.0),
            empirical_logits,
            fallback=-10.0,
        )
    if "LOGIT_RHO_L_SCALE" in stan_code:
        stan_data['LOGIT_RHO_L_SCALE'] = float(config['model'].get('logit_rho_l_scale', 2.0))
    if "LOGIT_RHO_H_LOCATION" in stan_code:
        stan_data['LOGIT_RHO_H_LOCATION'] = _resolve_logit_location(
            config['model'].get('logit_rho_h_location', -6.0),
            empirical_logits,
            fallback=-6.0,
        )
    if "LOGIT_RHO_H_SCALE" in stan_code:
        stan_data['LOGIT_RHO_H_SCALE'] = float(config['model'].get('logit_rho_h_scale', 2.5))
    if "W_RATE" in stan_code:
        stan_data['W_RATE'] = float(config['model'].get('w_rate', 0.5))
    if "K0_LOCATION" in stan_code:
        stan_data['K0_LOCATION'] = float(config['model'].get('k0_location', kmax / 2.0))
    if "K0_SCALE" in stan_code:
        stan_data['K0_SCALE'] = float(config['model'].get('k0_scale', kmax / 2.0))

    if is_vector_dev:
        stan_data['DEVIATION_SCALE'] = [dev_scale_val] * k_components
    elif "DEVIATION_SCALE" in stan_code:
        stan_data['DEVIATION_SCALE'] = dev_scale_val
        
    if is_jgy_mod:
        stan_data['REPLICATES'] = int(config['model'].get('replicates', 100))
    
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
        try:
            lp_values = np.asarray(fit.stan_variable("lp__"), dtype=float)
        except Exception:
            try:
                lp_values = np.asarray(fit.method_variables()["lp__"], dtype=float)
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
