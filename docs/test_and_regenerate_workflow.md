# Test & Data Regeneration Workflow

## Overview

This document describes how to:
1. Clean all processed data
2. Regenerate data using Snakemake
3. Run consolidated test suite
4. Verify everything works

## Test Suite Consolidation

### Old Tests (Redundant - Can Delete)
- `tests/test_author_reproducibility.py` → merged into `test_twitter_data.py`
- `tests/test_twitter_validation.py` → merged into `test_twitter_data.py`
- `tests/test_twitter_processing.py` → outdated, can delete
- `tests/test_data_processing.py` → split into `test_utils.py` and `test_oo_data.py`

### New Consolidated Tests
1. **`tests/test_twitter_data.py`** (MOST IMPORTANT)
   - Author reproducibility (compares with ground truth)
   - Intervention structure validation
   - "Full" aggregation correctness
   - Sanity checks

2. **`tests/test_oo_data.py`**
   - OO data processing
   - Contact adjacency tests
   - Infection time series tests

3. **`tests/test_utils.py`**
   - Melt/unmelt utilities
   - Save/load draws

## Workflow Steps

### Step 0: Clean Old Tests (Optional)

```bash
cd tests/

# Move old tests to archive
mkdir -p _archive
mv test_author_reproducibility.py _archive/
mv test_twitter_validation.py _archive/
mv test_twitter_processing.py _archive/
mv test_data_processing.py _archive/

# Keep only new consolidated tests
ls test_*.py
# Should show: test_twitter_data.py, test_oo_data.py, test_utils.py
```

### Step 1: Delete Processed Data

```bash
cd /Users/willthompson/Documents/CSDS/side_projects/heterogeneous-infection-kernels

# Delete all Twitter processed data
rm -rf data/processed/twitter/

# Delete OO processed data (optional - if you want to regenerate)
rm -rf data/processed/oo/

# Verify deletion
ls -la data/processed/
# Should be empty or only have .gitkeep
```

### Step 2: Regenerate Twitter Data with Snakemake

```bash
# Option A: Use Snakemake to regenerate q=1.0 data
snakemake data/processed/twitter/full/M_k_q1.0.parquet -j1

# This will:
# 1. Process all 8 interventions
# 2. Create M/N matrices for k and kappa
# 3. Aggregate into "full" directory

# Option B: Run the Python script directly
python3 src/data_processing/process_twitter.py --q 1.0

# For q=0.2 (if needed):
python3 src/data_processing/process_twitter.py --q 0.2
```

### Step 3: Verify Data Generation

```bash
# Check that all interventions were processed
ls data/processed/twitter/

# Should show:
# BanksySF/
# HowManyPushups/
# SFThanks/
# blackfridaystories/
# getyourflushot/
# highfiveastranger/
# somethinggood/
# turkeyface/
# full/  ← NEW!

# Check full aggregation
ls data/processed/twitter/full/

# Should show:
# M_k_q1.0.parquet
# N_k_q1.0.parquet
# M_kappa_q1.0.parquet
# N_kappa_q1.0.parquet

# Quick data check
python3 -c "
import pandas as pd
m = pd.read_parquet('data/processed/twitter/full/M_k_q1.0.parquet')
print(f'Full aggregation: {len(m)} users')
print(f'Total adoptions: {m.drop(columns=[\"user_id\"]).sum().sum():.0f}')
"
```

### Step 4: Run Consolidated Tests

```bash
# Run all tests with verbose output
uv run pytest tests/ -v -s

# Or run individual test files:

# 1. Twitter data tests (MOST IMPORTANT)
uv run pytest tests/test_twitter_data.py -v -s

# 2. Utility tests
uv run pytest tests/test_utils.py -v -s

# 3. OO tests (if OO data processed)
uv run pytest tests/test_oo_data.py -v -s
```

### Step 5: Verify Critical Test Passes

The **most important** test is author reproducibility:

```bash
uv run pytest tests/test_twitter_data.py::test_author_reproducibility -v -s
```

Expected output:
```
AUTHOR REPRODUCIBILITY TEST
======================================================================

Comparing our results with author's ground truth...

k     | Author M     | Our M        | Diff         | Match
-----------------------------------------------------------------
1     | 16.00        | 16.00        | 0.00         | ✓
2     | 6.00         | 6.00         | 0.00         | ✓
3     | 17.00        | 17.00        | 0.00         | ✓
...
-----------------------------------------------------------------
Total absolute difference: 0.00

✅ REPRODUCIBILITY TEST PASSED!
   Our pipeline matches author's results (total diff: 0.00)
```

If this test FAILS, the pipeline has a bug!

## Troubleshooting

### Problem: "Author ground truth not found"

**Solution**: The test looks for `data/twitter_contagion_data/twitter_contagion_code/ground_truth_q1.json`

You may need to generate this file from the author's code. If it doesn't exist, create it manually:

```python
# In Python
import json
import sys
from pathlib import Path

# Add author code to path
sys.path.insert(0, 'data/twitter_contagion_data/twitter_contagion_code')
import tools_p2 as tools  # or tools_p3

# Run author's code to get ground truth
result = tools.run_complex_contagion(q=1.0)

# Save as ground truth
with open('data/twitter_contagion_data/twitter_contagion_code/ground_truth_q1.json', 'w') as f:
    json.dump(result, f, indent=2)
```

### Problem: "Full aggregation doesn't match manual sum"

**Solution**: The `aggregate_all_interventions()` function might have an issue. Debug:

```python
import duckdb
con = duckdb.connect()

# Check what interventions exist
m_glob = "data/processed/twitter/*/M_k_q1.0.parquet"
files = con.query(f"SELECT filename FROM read_parquet('{m_glob}', filename=true) LIMIT 10").df()
print(files)

# Should NOT include 'full' in the glob!
# If it does, you're double-counting
```

### Problem: Tests run but reproducibility fails

**Solution**: Check data processing logic in `src/data_processing/process_twitter.py`:
- Verify exposure counting matches author's `compute_exposure_probabilities_fast()`
- Check that bot filtering (97.5th percentile) is applied
- Verify adoption detection (first retweet time) is correct

## Expected Test Results

All tests should pass:

```
tests/test_twitter_data.py::test_intervention_structure PASSED
tests/test_twitter_data.py::test_full_aggregation_exists PASSED
tests/test_twitter_data.py::test_full_aggregation_correctness PASSED
tests/test_twitter_data.py::test_author_reproducibility PASSED  ← CRITICAL
tests/test_twitter_data.py::test_sanity_checks PASSED

tests/test_utils.py::test_melt_unmelt_2d PASSED
tests/test_utils.py::test_melt_unmelt_3d PASSED
tests/test_utils.py::test_save_load_draws PASSED
tests/test_utils.py::test_melt_unmelt_1d PASSED
tests/test_utils.py::test_save_load_empty_dict PASSED

==================== 10 passed in 5.23s ====================
```

## After Tests Pass

Once all tests pass, you can:

1. **Run empirical Bayes inference**:
   ```bash
   uv run python scripts/run_twitter_full_inference.py --metric k --q 1.0
   ```

2. **Generate plots**:
   ```bash
   quarto render notebooks/will/twitter_full_results_FIXED.qmd
   ```

3. **Run Snakemake workflow** for full pipeline:
   ```bash
   snakemake -j4
   ```
