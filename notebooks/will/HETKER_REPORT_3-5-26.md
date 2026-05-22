# HetKer Project Report — 3-5-26

## Summary of Work (3-2-26 to 3-5-26)

---

### 1. Pipeline Validation and Testing

Wrote `test/test_pipeline_vs_reproduce.py` — 16 unit tests comparing pipeline M/N parquet output against the Monsted author replication script at production q values (q_sc=0.192, q_cc=0.18). All passing within 1pp for adoption probability. Validated against hand-transcribed Monsted Fig 3 bar heights.

---

### 2. Dashboard Overhaul

Rewired `dashboard.qmd` to read from pipeline M/N parquets via DuckDB instead of raw on-the-fly binomial smoothing. Fixed a cascade of bugs:

- 100x scale mismatch between Monsted tab and Main Dashboard tab
- NaN `m_name` crash (stan_file param missing from optimal-q YAML configs)
- NaN `y_max` crash on ylim
- Model overlay SQL bug (`mu%` matching latent `c` rows — changed to `mu[%`)
- Kernel reconstruction fallback when Stan generated quantities have NULL mu (due to log-overflow in original Stan code — `log(c)` instead of `log_inv_logit(alpha)`)
- Jupyter kernel fix: switched from python3 (base conda, broken scipy) to hetker (.venv Python 3.13)

Added an estimator dropdown (mean / median / marginal MAP / joint MAP) to kernel overlay. All models Playwright-verified.

**Dashboard launch:** `QUARTO_PYTHON=.venv/bin/python3 quarto preview dashboard.qmd --port 4242`

---

### 3. New Inference Sweeps at Optimal q

Created two new sweep configs at production q values:

| Sweep | q | Metric | K | Models | Jobs |
|-------|---|--------|---|--------|------|
| `research_sweep_twitter_optimal_q_sc` | 0.192 | k | 1, 2 | SC, CC, NP mono | 6 |
| `research_sweep_twitter_optimal_q_cc` | 0.18 | kappa | 1, 2 | SC, CC, NP mono | 6 |

All 12 jobs completed on VACC and synced locally. Results in `results/research_sweep_twitter_optimal_q_sc/` and `results/research_sweep_twitter_optimal_q_cc/`.

---

### 4. Model Reparameterization (Critical Fix)

**Root cause of scale mismatch identified:**

- NP model: simplex constraint forces mu to sum to 1 across kmax=20 levels (~0.05 each), but true Twitter adoption rates are ~0.0002. Simplex is structurally wrong — adoption probabilities per exposure level do not sum to 1.
- Parametric: Beta(1,10) prior on rho has E=0.091, but data needs rho~0.00015 — prior is 607x too high.

**Fix: logit-Normal reparameterization for all models**

| Model | Parameter | Prior | Posterior median |
|-------|-----------|-------|-----------------|
| SC | rho | Normal(-10, 2) on logit | ~4.5e-5 |
| CC | rho_l | Normal(-10, 2) on logit | ~4.5e-5 |
| CC | rho_h | Normal(-6, 2.5) on logit | ~2.5e-3 |
| NP mono | logit_p location | -7.5 | ~5.5e-4 |

All 3 sweeps (twitter, optimal_q_sc, optimal_q_cc) rerun on VACC with fixed priors.

**Stan model fixes also applied:**
- Added `log_lik` to generated quantities for WAIC computation
- Fixed overflow: `log(c)` → `log_inv_logit(alpha)` in generated quantities (was producing NaN for all mu when sigma~25)
- NP non-monotonic dropped from all configs (not well-motivated scientifically)

**Joint MAP estimator:** NaN due to individual draw overflow in `c` computation — cosmetic, other 3 estimators (mean, median, marginal MAP) all work.

---

### 5. NP Mono Sigma Prior Fix (OO Dataset)

OO dataset (Sim 165, backbone disparity): NP mono sigma was blowing up to 27 with Exp(1) prior — model was fitting per-user heterogeneity instead of kernel shape.

**Fix:** Changed `Exp(1)` to `HalfNormal(0, 1)` in `non_parametric_monotonic_empirical_bayes.stan`.

| | K=1 | K=2 |
|-|-----|-----|
| sigma (old, Exp(1)) | ~27 | ~27 |
| sigma (new, HalfNormal) | 1.50 (95% CI: 1.28–1.72) | 0.53 (95% CI: 0.11–0.92) |

**kmax=30 boundary test:** The convex uptick at nu=20 (kmax=20) was a boundary artifact — kmax=30 kernel is smooth and approximately linear/gently concave. No evidence of complex contagion in OO Sim 165. SC model is adequate.

---

### 6. Open Issues / Deferred Items

- **NP mono sigma still needs checking on Twitter data** — prior fix applied to OO sweep only; confirm sigma is reasonable in optimal_q Twitter runs
- **q mismatch for fast-path loading:** Pipeline parquets use q=0.192/0.18 but inference runs use q=0.2/1.0. Dashboard has fallback (on-the-fly computation) but fast path not working everywhere. Fix: rerun pipeline at q=0.2/1.0 or rerun inference at q=0.192/0.18.
- **WAIC values:** Need to verify log_lik is correctly computed in all Stan models post-fix. Check inference_summary.parquet for elpd_waic column.
- **Parametric K=1 models:** mu values were NULL pre-fix — verify they are now finite after Stan overflow fix.

---

### 7. Agent Instructions for Twitter Sweep Notebook (3-5-26.qmd)

**Notebook location:** `notebooks/will/3-5-26.qmd`
**Environment:** `QUARTO_PYTHON=.venv/bin/python3 quarto render notebooks/will/3-5-26.qmd`
**Kernel:** `hetker`

#### Section 1 — Setup
Standard imports: duckdb, pandas, numpy, matplotlib, pathlib.
Set `RESULTS_ROOT = Path("../../results")` from notebook location.
Focus on `research_sweep_twitter_optimal_q_sc` and `research_sweep_twitter_optimal_q_cc` — these are the primary results with corrected priors.

#### Section 2 — Empirical Probability of Adoption
Load M and N parquets for both q_sc=0.192 (k metric) and q_cc=0.18 (kappa metric):
```
data/processed/twitter/full/M_k_q0.192.parquet
data/processed/twitter/full/N_k_q0.192.parquet
data/processed/twitter/full/M_kappa_q0.18.parquet
data/processed/twitter/full/N_kappa_q0.18.parquet
```
Compute `P_adopt(k) = 100 * M[k] / (M[k] + N[k])` (percentage).
Plot: bar chart of P(adopt|k) vs k for both metrics, with binomial 95% CI error bars.
Save to `figures/3-5-26/empirical_adoption.png`.

#### Section 3 — N and M Matrices
Side-by-side bar charts of M[k] and N[k] for both metrics. Log scale y-axis. Include total counts.
Save to `figures/3-5-26/M_N_matrices.png`.

#### Section 4 — Kernel Fits
Load `kernel_mode_summary.parquet` from all optimal_q runs. Columns include: `nu`, `mean`, `median`, `map_marginal`, plus path-derived model identifiers.

For each (sweep × K) combination, plot all 3 models (SC, CC, NP mono) overlaid. Use `mean` estimator as primary curve. Overlay empirical P(adopt|k) in light gray background.

Layout: 2×2 grid (rows: SC metric / CC metric, cols: K=1 / K=2).
Save to `figures/3-5-26/kernel_fits.png`.

#### Section 5 — WAIC Comparison
Load `inference_summary.parquet` from all optimal_q runs. Check for `elpd_waic`, `waic`, or `looic` columns. Compute ΔWAIC (subtract best model within each group; lower = better). Bar chart faceted by K=1 / K=2, colored by model. Annotate bars with ΔWAIC values.
Save to `figures/3-5-26/waic_comparison.png`.

#### Section 6 — Summary Table
DataFrame with all runs: sweep, K, model, q, metric, WAIC, rho_mean, rho_sd, sigma_mean (NP only). Display as styled table, best model highlighted per group.

#### Notes for agent
- NP non-monotonic was dropped from optimal_q sweeps — do not expect it
- `research_sweep_twitter` uses old Beta priors — focus on optimal_q sweeps
- Joint MAP estimator is NaN — use `mean` or `map_marginal`
- `dashboard.qmd` has working data loading code to reference
- Check parquet column names before assuming — read one file with `pd.read_parquet(path).columns.tolist()`
