# Research Workflow Instructions

## Running Sweeps on VACC

We use `just` to orchestrate research tasks, leveraging `ssh-mux` for persistent connections to the VACC cluster.

### 1. Data Generation
To generate processed data (Simulated or Twitter):
```bash
~/.tools/bin/ssh-mux exec vacc -- "cd CSDS/side_projects/heterogeneous-infection-kernels && uv run snakemake -c1 <target_file>"
```
Example: `data/processed/oo/sim_165_backbone_disparity/exposures.parquet`

### 2. Running Research Sweeps
We use Snakemake profiles to manage SLURM job submissions.

**Command:**
```bash
~/.tools/bin/ssh-mux exec vacc -- "cd CSDS/side_projects/heterogeneous-infection-kernels && uv run snakemake inference --configfile configs/<config_name>.yaml --profile workflow/profiles/slurm"
```

*   **Config File:** YAML file in `configs/` defining the sweep parameters (e.g., `research_sweep_oo.yaml`).
*   **Profile:** `workflow/profiles/slurm` handles job submission, resource requests (partition `week`), and logging.

### 3. Monitoring Jobs
Check job status on VACC:
```bash
~/.tools/bin/ssh-mux exec vacc -- "squeue -u wthomps3"
```

### 4. Syncing Results
Pull results from VACC to local machine:
```bash
just sync
```
(This runs `rsync` from `vacc:.../results/` to `results/`)

---
## Environment Notes
- **Python:** Managed via `uv`. Always prefix commands with `uv run`.
- **Connections:** Use `~/.tools/bin/ssh-mux` for all VACC commands to reuse the SSH socket.
