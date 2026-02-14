# Inference Workflow Setup Checklist

## Overview
This document tracks what's needed for a complete Snakemake inference workflow with flexible data and model selection.

## Requirements

### 1. Data Loading
- [x] **Twitter data processing** - `src/data_processing/process_twitter.py` generates M/N parquet files
- [ ] **Twitter aggregation across interventions** - Need function to join M/N on user_id and sum
- [x] **Operation Outbreak data** - `src/data_processing/operation_outbreak.py` processes OO sims
- [ ] **Config-driven data loading** - Read data source from config and load appropriately

### 2. Model Configuration
- [x] **Config templates** - `configs/inference_template.yaml` defines structure
- [x] **Empirical Bayes support** - Models updated with `LOGIT_P_LOCATION` parameter
- [ ] **Auto-compute empirical prior** - Script needs to detect "empirical" and compute from data
- [ ] **Model registry** - Validate model names against available .stan files

### 3. Inference Script Updates

#### Current State:
- `scripts/run_inference.py` exists but:
  - Loads data from "exposure_summary.parquet" (old format)
  - Doesn't handle intervention aggregation
  - Doesn't compute empirical Bayes parameters
  - Uses random sampling for probabilistic exposure (not parquet M/N)

#### Needed Updates:
- [ ] **Data loading**:
  ```python
  def load_data_from_config(config):
      if config['data']['source'] == 'twitter':
          if config['data']['twitter']['interventions'] == 'ALL':
              # Load M/N parquets from all interventions
              # Join on user_id and SUM across interventions
              return aggregate_twitter_all_interventions(...)
          else:
              # Load specific intervention(s)
              return load_twitter_interventions(...)
      elif config['data']['source'] == 'oo':
          return load_oo_simulation(...)
  ```

- [ ] **Empirical Bayes computation**:
  ```python
  def compute_empirical_prior(M, N, k_low=5):
      m_total = M.sum(axis=0)
      n_total = N.sum(axis=0)
      rho_empirical = np.sum(m_total[:k_low]) / (np.sum(m_total[:k_low] + n_total[:k_low]) + 1e-10)
      return logit(rho_empirical)
  ```

- [ ] **Model instantiation from config**:
  ```python
  for model_config in config['models']:
      stan_data = prepare_stan_data(M, N, model_config)
      if 'LOGIT_P_LOCATION' in stan_data and stan_data['LOGIT_P_LOCATION'] == 'empirical':
          stan_data['LOGIT_P_LOCATION'] = compute_empirical_prior(M, N)
      fit = run_stan_model(model_config['stan_file'], stan_data, model_config['sampling'])
  ```

### 4. Snakemake Integration

#### Current State:
- `workflow/rules/inference.smk` exists
- Expects: `configs/{sweep_name}.yaml` → `results/{sweep_name}/{data_source}/{data_name}/{instance_id}/`

#### Needed Updates:
- [ ] **Update inference rule** to use new config system
- [ ] **Add data dependency resolution**:
  ```python
  def get_data_inputs(wildcards):
      config = load_config(f"configs/{wildcards.config_name}.yaml")
      if config['data']['source'] == 'twitter':
          if config['data']['twitter']['interventions'] == 'ALL':
              return expand("data/processed/twitter/{intervention}/M_{metric}_q{q}.parquet",
                           intervention=INTERVENTIONS, ...)
          else:
              return [f"data/processed/twitter/{i}/M_{metric}_q{q}.parquet" for i in ...]
      else:  # OO
          return f"data/processed/oo/sim_{sim}_backbone_{method}_exposures.parquet"
  ```

- [ ] **Wildcard from config**:
  ```python
  rule run_inference_from_config:
      input:
          data = get_data_inputs,
          config = "configs/{config_name}.yaml"
      output:
          draws = "results/{config_name}/{model_name}/draws/",
          summary = "results/{config_name}/{model_name}/summary.csv"
      shell:
          "uv run python scripts/run_inference_from_config.py --config {input.config} --model {wildcards.model_name} --output {output.draws}"
  ```

### 5. Twitter Data Aggregation Function

This is CRITICAL and currently missing:

```python
# In src/data_processing/utils.py or new file
def aggregate_twitter_all_interventions(metric='k', q=1.0):
    """
    Load M and N matrices from all interventions and aggregate by user_id.

    Returns:
        M: [V, kmax] array where M[i,k] = sum across interventions of M_intervention[i,k]
        N: [V, kmax] array (same)
        user_ids: [V] array of user IDs
    """
    import duckdb
    con = duckdb.connect()

    data_dir = Path("data/processed/twitter")
    m_files = str(data_dir / f"*/M_{metric}_q{q}.parquet")
    n_files = str(data_dir / f"*/N_{metric}_q{q}.parquet")

    # Get column names
    cols_check = con.query(f"SELECT * FROM '{m_files}' LIMIT 0").df().columns
    cols_data = [c for c in cols_check if c not in ['user_id', 'intervention']]

    # Aggregate across interventions per user
    sum_m_expr = ", ".join([f"SUM(\"{c}\") as m_{c}" for c in cols_data])
    sum_n_expr = ", ".join([f"SUM(\"{c}\") as n_{c}" for c in cols_data])

    m_agg = con.query(f"SELECT user_id, {sum_m_expr} FROM '{m_files}' GROUP BY user_id ORDER BY user_id").df()
    n_agg = con.query(f"SELECT user_id, {sum_n_expr} FROM '{n_files}' GROUP BY user_id ORDER BY user_id").df()

    user_ids = m_agg['user_id'].values
    M = m_agg.drop(columns=['user_id']).values
    N = n_agg.drop(columns=['user_id']).values

    return M, N, user_ids
```

### 6. Testing

- [ ] **Test config parsing** - Load config and validate all fields
- [ ] **Test data loading** - Verify Twitter ALL aggregation gives same result as `run_twitter_full_inference.py`
- [ ] **Test empirical Bayes** - Verify computed prior matches `compute_empirical_prior.py` output
- [ ] **Test single model run** - Run one model end-to-end from config
- [ ] **Test full workflow** - Snakemake dry-run and actual run

## Action Plan

1. **Immediate** (to make it work now):
   - Copy aggregation logic from `run_twitter_full_inference.py` into `src/data_processing/utils.py`
   - Update `run_inference.py` to use new config format
   - Add empirical prior computation

2. **Short-term** (clean up):
   - Refactor Snakemake rules to use new system
   - Add validation for configs
   - Add logging and error handling

3. **Long-term** (nice to have):
   - Add config schema validation (e.g., with `jsonschema`)
   - Add model comparison utilities
   - Add automatic BIC/WAIC computation

## Current Working Solution

For NOW, you can use `scripts/run_twitter_full_inference.py` directly (already updated with empirical Bayes):

```bash
# Run with updated empirical Bayes priors
uv run python scripts/run_twitter_full_inference.py --metric k --q 1.0
```

This will use the corrected priors and save results to `results/twitter_full_k_q1.0/`.

## Files to Create/Update

### New Files Needed:
- [ ] `src/data_processing/load_data.py` - Unified data loading interface
- [ ] `scripts/run_inference_from_config.py` - Config-driven inference script
- [ ] `scripts/validate_config.py` - Config validation utility

### Files to Update:
- [ ] `scripts/run_twitter_full_inference.py` - Extract aggregation function
- [ ] `scripts/run_inference.py` - Align with new config system or deprecate
- [ ] `workflow/rules/inference.smk` - Update to use new config system
- [ ] `src/data_processing/utils.py` - Add aggregation function

## Questions to Resolve

1. **Config scope**: Should one config file specify multiple models (current template) or one model per config?
   - **Recommendation**: Multiple models per config (allows comparison)

2. **Results directory structure**: Current Snakemake uses `results/{sweep_name}/{data_source}/{data_name}/{instance_id}/`
   - **Recommendation**: Simplify to `results/{config_name}/{model_name}/`

3. **Backward compatibility**: Keep old `run_inference.py` or replace entirely?
   - **Recommendation**: Keep for now, add deprecation warning

4. **Model selection in Snakemake**: How to expand over models in config?
   - **Recommendation**: Add checkpoint to read config and generate model targets dynamically
