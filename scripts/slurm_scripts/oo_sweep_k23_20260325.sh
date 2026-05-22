#!/bin/bash
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=oo_sweep_k23_20260325
#SBATCH --output=/gpfs1/home/w/t/wthomps3/CSDS/side_projects/heterogeneous-infection-kernels/logs/oo_sweep_k23_20260325_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=will.w.thompson@gmail.com

cd /gpfs1/home/w/t/wthomps3/CSDS/side_projects/heterogeneous-infection-kernels
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH

echo "Running on host: $(hostname)"
echo "Start time: $(date)"
echo "OO Sim 165 kmax30 K=2,3 model selection -- March 25 2026"

mkdir -p logs

uv run snakemake inference \
  --configfile configs/oo_sweep_k23_20260325.yaml \
  --profile workflow/profiles/local \
  --cores 4 \
  --rerun-incomplete

echo "End time: $(date)"
echo "Exit code: $?"
