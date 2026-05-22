#!/bin/bash
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=oo_np_k23_rerun_20260325
#SBATCH --output=/gpfs1/home/w/t/wthomps3/CSDS/side_projects/heterogeneous-infection-kernels/logs/oo_np_k23_rerun_20260325_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=will.w.thompson@gmail.com

cd /gpfs1/home/w/t/wthomps3/CSDS/side_projects/heterogeneous-infection-kernels
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH

echo "Running on host: $(hostname)"
echo "Start time: $(date)"
echo "OO NP K=2/3 rerun with log_lik fix -- March 25 2026"

mkdir -p logs

uv run snakemake inference \
  --configfile configs/oo_np_k23_rerun_20260325.yaml \
  --profile workflow/profiles/local \
  --cores 4 \
  --rerun-incomplete \
  --forcerun run_inference

echo "End time: $(date)"
echo "Exit code: $?"
