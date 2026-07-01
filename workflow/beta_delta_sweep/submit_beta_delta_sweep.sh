#!/bin/bash
#SBATCH --job-name=sweep_beta_delta
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/sweep_beta_delta_%A_%a.out
#SBATCH --error=logs/sweep_beta_delta_%A_%a.err
#SBATCH --mail-user=kbt28@cornell.edu
#SBATCH --mail-type=END

OFFSET=${OFFSET:-0} # defaults to 0 if not passed
REAL_ID=$((SLURM_ARRAY_TASK_ID + OFFSET))

module load python/3.11.5

cd ~/het-ker

# args: <manifest>  <task_id from array index>  <results_root>
uv run python3 workflow/run_task.py manifests/inference_beta_delta_sweep.jsonl "${REAL_ID}" results
