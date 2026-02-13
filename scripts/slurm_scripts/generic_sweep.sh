#!/bin/bash
#SBATCH --partition=week
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00

# Load modules if needed (e.g., git, python if not using uv directly)
# module load python/3.11-2024-01

# Navigate to repo root
cd $SLURM_SUBMIT_DIR

# Ensure uv is in path or actviated
# Assuming ~/.cargo/bin is in PATH or similar
export PATH=$HOME/.cargo/bin:$PATH

echo "Running on host: $(hostname)"
echo "Repo root: $(pwd)"

