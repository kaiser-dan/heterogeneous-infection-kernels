# TODO

## Current Priority

- Finish the Snakemake cleanup first.
- Tighten the `analysis.ppc` workflow wiring before enabling it broadly.
- Resolve summary-format mismatches in the PPC path.
  The PPC scripts expect a summary directory containing `M.parquet` and `N.parquet`.
  The current Twitter summary rule writes a single merged parquet instead, so Twitter PPC is not wired end-to-end yet.
- Generalize the PPC pairing/config surface if we want more than the current simple-vs-nonparam comparison.
  Right now `workflow/lib/experiment_plan.py` only constructs one PPC pair:
  `model_simple_contagion` vs `non_parametric_monotonic_empirical_bayes`.

## Deferred: Mahalanobis / Complexity

- Revisit the Mahalanobis non-centrality statistic after the SMK work is done.
- Check whether replacing the current mean-based offset with a mode-centered offset makes sense under multimodal posteriors.
  The current implementation in `src/complexity_measures/malahbois.py` uses the posterior mean kernel and the mean simple-model `rho`.
- If we change the location summary, preserve the interpretation of `lambda` as an offset-from-manifold statistic.
  "Average distance to the mode" is not the same object as the current non-centrality parameter and may mix signal with posterior spread.
- If multimodality is real, consider cluster-aware or MAP/mode-centered variants instead of a plain posterior mean.

## Deferred: Posterior Predictive Checks

- Fully implement the PPC checks after the Snakemake path is stable.
- Decide which PPC statistics are canonical for each use case:
  non-nested fit diagnostics,
  nested re-fit calibration,
  and complexity / manifold-separation checks.
- Add broader end-to-end workflow execution coverage for PPC once the full fitted-data path is stable.
