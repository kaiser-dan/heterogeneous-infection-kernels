# Proposal: Durable Synthetic Sweep Pipeline for Inference

Date: 2026-02-14  
Project: `heterogeneous-infection-kernels`

## 1) Executive Summary

This proposal introduces a durable synthetic-data pipeline that:

1. Generates and stores simulated datasets under `data/processed/synthetic/`.
2. Uses stable Parquet artifacts (`M`, `N`, metadata, graph artifacts) so simulation is decoupled from inference.
3. Integrates with current Snakemake inference by reusing the same data-loading pattern as existing processed data.
4. Supports parameter sweeps over graph generation and contagion dynamics without rerunning simulations unnecessarily.
5. Enables full DuckDB querying over synthetic simulation metadata and results.

Core idea: **simulate once, fit many times**.

## 2) Goals

1. Add first-class support for synthetic/simulated datasets in the pipeline.
2. Keep inference workflow consistent with current OO/Twitter ingestion.
3. Make synthetic data queryable and discoverable with DuckDB + Parquet.
4. Ensure reproducibility (seeded, versioned, immutable run outputs).
5. Avoid repeated simulation for each inference model sweep.

## 3) Non-Goals

1. Replacing all existing OO/Twitter rules immediately.
2. Building every possible graph kernel assignment strategy in phase 1.
3. Fully redesigning current inference code in one pass.

## 4) Current State (Relevant Observations)

1. Current `Snakefile` expands `config.sweep` for inference outputs but assumes a single `DATA_CFG_NAME` source target.
2. Data prep rules currently focus on Twitter and OO processing.
3. Inference already has an OO pathway that loads precomputed `M.parquet` and `N.parquet`.
4. This OO behavior is a good template for synthetic integration.

Conclusion: we can add a synthetic source with minimal disruption by emitting OO-like artifacts and routing inference to them.

## 5) Proposed Data Model

## 5.1 Directory Layout

Store synthetic outputs under:

`data/processed/synthetic/<sim_id>/`

Where `<sim_id>` is deterministic from parameters (human-readable + short hash), for example:

`graph~sbm/n~256/k~18/epsilon~0.65/contagion~mixed_simple/omega~0.10/gamma~0.10/tmax~10000/seed~37/hash~9f2a1c`

Phase-1 required files:

1. `M.parquet`
2. `N.parquet`
3. `edges.parquet` (or `adjacency.parquet` if preferred; pick one and standardize)
4. `states.parquet` (optional in phase 1, recommended for diagnostics)
5. `kernels.parquet` (node-level kernel assignment and kernel values or refs)
6. `metadata.parquet` (single-row table for DuckDB)
7. `metadata.json` (exact config snapshot for reproducibility/audit)

No global index is required in phase 1.
DuckDB can query recursive `metadata.parquet` files directly from the directory tree.

## 5.2 Metadata Columns (Minimum)

1. `sim_id`
2. `graph_type` (`sbm`, `er`, `zkc`)
3. `n`
4. graph params (`epsilon`, `avg_degree`, etc.)
5. `contagion_type` (`simple`, `sigmoid`, `nonparametric`, `mixture`)
6. kernel assignment mode (`block_flip`, `iid`, `explicit`)
7. `omega`
8. `gamma`
9. `tmax`
10. `kmax`
11. `seed`
12. `code_version` (git commit if available)
13. `created_at`

## 6) Config Design

Use two sweep namespaces:

1. `sim_sweep`: synthetic generation parameters
2. `sweep`: inference model parameters (existing behavior)

Example:

```yaml
name: "synthetic_sbm_r0"
source: "synthetic"

synthetic:
  graph:
    type: "sbm"
    params:
      n: 256
      avg_degree: 18
      epsilon: 0.9
  contagion:
    assignment:
      mode: "block_flip"
      omega: 0.1
    kernels:
      - id: "k1"
        type: "simple"
        beta: 0.01
      - id: "k2"
        type: "simple"
        beta: 0.04
    gamma: 0.1
    tmax: 10000
  seed: 37

sim_sweep:
  synthetic.graph.params.epsilon: [0.05, 0.15, 0.25, 0.35]
  synthetic.contagion.assignment.omega: [0.0, 0.1, 0.2, 0.3]
  synthetic.contagion.kernels.0.beta: [0.01]
  synthetic.contagion.kernels.1.beta: [0.04]

sweep:
  K: [1, 2]
  stan_file:
    - "src/models/non_parametric_monotonic.stan"
    - "src/models/model_simple_contagion.stan"
```

Notes:

1. Dot-path keys in `sim_sweep` allow sweeping nested fields.
2. Keep `sweep` unchanged so existing inference expansion remains familiar.

## 7) Graph and Contagion Abstractions

## 7.1 Graph Generators (`src/networks.py`)

Add or standardize:

1. `sbm(n, k, epsilon, seed)`
2. `er(n, p=None, k=None, seed)` where one of `p` or `k` is provided
3. `zkc()` for Zachary Karate Club

Return a consistent adjacency format (numpy array preferred).

## 7.2 Kernel Specification

Support kernel definitions:

1. `simple`: `beta`
2. `sigmoid`: `rho_l`, `rho_h`, `w`, `k0`
3. `nonparametric`: explicit vector for `nu=1..kmax`

## 7.3 Node-Kernel Assignment

Support assignment strategies:

1. `block_flip` (best first target for SBM):
   - initialize by block
   - flip block kernel label with probability `omega`
2. `iid`:
   - sample kernel id from global probabilities
3. `explicit`:
   - load mapping `node_id -> kernel_id` from file

## 8) Snakemake Integration Plan

## 8.1 New Rules File

Add:

`workflow/rules/simulation.smk`

Core rules:

1. `generate_synthetic_sim`:
   - input: config
   - output: `data/processed/synthetic/{sim_id}/M.parquet`, `N.parquet`, etc.
   - script: `workflow/scripts/simulate/generate_sim_data.py`
2. (Optional, later) `register_synthetic_manifest` if run discovery becomes slow at scale.

## 8.2 Snakefile Source Routing

Add `source == "synthetic"` branch:

1. Build synthetic data target names from `sim_sweep` expansion.
2. Link inference inputs to generated synthetic directories.
3. Expand final inference outputs over both simulation instances and inference `sweep`.

This makes workflow:

`sim param combo -> synthetic artifacts -> inference sweep combo`

## 8.3 Inference Compatibility

Minimal change path:

1. Reuse OO-style loading path for `M/N` from synthetic directory.
2. In `src/models/inference.py`, treat `source in {"oo", "synthetic"}` similarly.
3. Use synthetic `M/N` directly; skip any dynamic discretization fallback.

## 9) DuckDB Query Model

With partitioned Parquet metadata, example queries:

```sql
SELECT sim_id, graph_type, epsilon, omega, gamma, tmax
FROM read_parquet(
  'data/processed/synthetic/**/metadata.parquet',
  hive_partitioning = true
)
WHERE graph_type = 'sbm' AND epsilon BETWEEN 0.2 AND 0.8;
```

Join synthetic metadata with inference outputs by `sim_id` and instance path tags (or explicit metadata columns in result parquet).

Recommendation:

1. Persist a small `run_metadata.parquet` alongside each inference output with parsed instance params.
2. This makes model-result joins in DuckDB trivial.

## 10) Reproducibility and Caching

1. Deterministic `sim_id`: canonical JSON of sim params + seed -> hash.
2. Save full config snapshot per run (`metadata.json`).
3. Include `code_version` when possible (`git rev-parse HEAD`).
4. Snakemake naturally avoids reruns when outputs exist and inputs are unchanged.
5. Add optional `--force-resim` pathway by deleting selected sim directories.

## 11) Performance Considerations

1. Simulation and inference should be separate DAG stages.
2. Simulation can be parallelized over `sim_sweep` combinations.
3. Inference can fan out over model sweeps for each precomputed synthetic dataset.
4. Keep intermediate arrays in Parquet or NPY depending on size:
   - `M/N` as Parquet (query-friendly)
   - large dense states may optionally be NPY + summarized Parquet metadata.

## 12) Validation and Tests

Add tests for:

1. Config expansion correctness for nested `sim_sweep`.
2. Deterministic `sim_id` generation.
3. Synthetic artifact schema checks (`M/N` shapes, metadata fields present).
4. Inference read-path on synthetic data.
5. End-to-end dry run DAG generation for a small synthetic config.

## 13) Rollout Plan

Phase 1 (vertical slice):

1. `source: synthetic` support.
2. `sbm` graph only.
3. kernels: `simple`, `sigmoid`, `nonparametric`.
4. assignment: `block_flip`.
5. inference integration via existing `M/N` route.

Phase 2:

1. `er`, `zkc`.
2. assignment: `iid`, `explicit`.
3. richer diagnostics and plots.

Phase 3:

1. optional synthetic result warehouse views in DuckDB.
2. optimization and larger experiment presets.

## 14) Risks and Mitigations

1. Risk: naming/path explosion in multi-dimensional sweeps.
   - Mitigation: hashed canonical `sim_id` plus compact human-readable tags.
2. Risk: duplicated logic between OO and synthetic loaders.
   - Mitigation: shared `load_precomputed_mn(directory)` helper.
3. Risk: unclear schema drift over time.
   - Mitigation: versioned metadata field `schema_version`.
4. Risk: over-coupling simulation config and inference config.
   - Mitigation: keep `sim_sweep` and `sweep` separate by design.

## 15) Acceptance Criteria

1. User can run one command and generate synthetic + inference sweeps.
2. Synthetic data is stored durably under `data/processed/synthetic`.
3. Re-running inference on same synthetic data does not re-run simulation.
4. DuckDB query can filter synthetic runs by graph/contagion parameters.
5. Existing Twitter/OO workflows continue to function.

## 16) Open Questions for Review

1. Preferred adjacency artifact name: `edges.parquet` vs `adjacency.parquet`?
2. Keep `states` by default or make optional to reduce storage?
3. Do we need a manifest table later, or can we stay purely filesystem + DuckDB?
4. Should inference outputs include explicit parsed columns (`sim_id`, `stan_file`, `K`) in a sidecar metadata parquet?
5. Do we want to standardize synthetic directory naming to exact OO-style (`sim_<id>_...`) for maximal compatibility?
