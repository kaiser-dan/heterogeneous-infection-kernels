#!/bin/bash
#SBATCH --partition=week
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --job-name=cv_model_sel_20260501
#SBATCH --output=logs/cv_model_sel_20260501_%j.out
#SBATCH --error=logs/cv_model_sel_20260501_%j.err

cd $SLURM_SUBMIT_DIR
export PATH=$HOME/.cargo/bin:$PATH

echo "Running on host: $(hostname)"
echo "Repo root: $(pwd)"

mkdir -p logs

echo "=== K1 ground truth sweep ==="
uv run snakemake cv_aggregate \
    --configfile configs/cv_model_sel_k1_gt_20260501.yaml \
    --profile workflow/profiles/slurm

echo "=== K2 ground truth sweep ==="
uv run snakemake cv_aggregate \
    --configfile configs/cv_model_sel_k2_gt_20260501.yaml \
    --profile workflow/profiles/slurm

echo "=== All sweeps complete ==="
