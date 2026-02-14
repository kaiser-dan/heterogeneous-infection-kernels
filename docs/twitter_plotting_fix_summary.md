# Twitter Results Plotting - Issues & Fixes

## Issues Found in `twitter_full_results.qmd`

### 1. **Non-Parametric Models Not Shown**
**Problem**: Lines 241 & 246 pass `None, None` for NP models instead of loading them

**Original Code (Line 241)**:
```python
plot_panel(axes[0, 0], sc_k['m'], sc_k['n'], sc_k, None, None, xmax=15, label_x="k")
```

**Should Be**:
```python
plot_panel(axes[0, 0], sc_k['m'], sc_k['n'], sc_k,
           mono_k, non_mono_k, xmax=15, label_x="k")
```

**Why It Matters**: Users can't compare non-parametric (monotonic/non-monotonic) with parametric models

---

### 2. **Non-Parametric Models Only Loaded for k, Not κ**
**Problem**: Lines 171-174 load NP models, but they're loaded for BOTH metrics now

**Current State**: NP model directories exist for both:
- `results/twitter_full_k_q{q}/` - k metric
- `results/twitter_full_kappa_q{q}/` - κ metric

**Fixed**: Load both in the corrected version

---

### 3. **Potential q=0.2 Simple Contagion Mismatch**
**User Concern**: "Author parametric for q=0.2 is incorrect and does not match the data at all"

**Hypothesis**: The q=0.2 simple contagion model might be using q=1.0 kernel

**How to Check**:
1. Load both `simple_k_q0.2_results.npz` and `simple_k_q1.0_results.npz`
2. Compare the `c` arrays (inferred kernels)
3. They should be DIFFERENT (q=0.2 has more smoothing)
4. If they're nearly identical → bug in model generation

---

## Fixed Notebook: `twitter_full_results_FIXED.qmd`

### Features:
1. ✅ **Verification Section**: Checks if q=0.2 ≠ q=1.0 for simple contagion
2. ✅ **All Models Plotted**: Parametric + NP Monotonic + NP Non-Monotonic
3. ✅ **Both Metrics**: Works for k (simple) and κ (complex)
4. ✅ **Diagnostic Tools**: RMSE check for parametric fit quality

### Plot Layout:
```
Row A (Probability):
  [SC k - All Models]  [CC κ - All Models]

Row B (Counts):
  [SC k - All Models]  [CC κ - All Models]
```

Each panel shows:
- Gray bars: Observed data
- Black circles: Parametric (Author)
- Blue squares: NP Monotonic
- Red triangles: NP Non-Monotonic

---

## Next Steps

### 1. Run the Fixed Notebook
```bash
cd notebooks/will
quarto preview twitter_full_results_FIXED.qmd
```

### 2. Check Output
**If q=0.2 parametric line is way off:**
- Verify the data files: `results/author_models/simple_k_q0.2_results.npz`
- Check the script that generated them
- May need to re-run inference

**If q=0.2 matches q=1.0:**
- Bug in the script that generates parametric models
- Need to fix and re-run

### 3. Compare Models
Once plots are correct, assess:
- Do NP models fit better than parametric?
- Is there evidence for monotonicity?
- Does complex contagion (κ) outperform simple (k)?

---

## File Locations

### Data Files
- Parametric: `results/author_models/simple_k_q{q}_results.npz`
- NP (k): `results/twitter_full_k_q{q}/monotonic_K1_draws/`
- NP (κ): `results/twitter_full_kappa_q{q}/monotonic_K1_draws/`

### Scripts
- Original notebook: `notebooks/will/twitter_full_results.qmd`
- Fixed notebook: `notebooks/will/twitter_full_results_FIXED.qmd`
- Generate parametric: `scripts/run_author_models_inference.py` (probably)

---

## Key Insight

The user wants to see **ALL models on the same plot** to compare:
1. Simple vs Complex contagion (k vs κ)
2. Parametric vs Non-Parametric
3. Monotonic vs Non-Monotonic

This will reveal which modeling choices matter most for explaining the Twitter data.
