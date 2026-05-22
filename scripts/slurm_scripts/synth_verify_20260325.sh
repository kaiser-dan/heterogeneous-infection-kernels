#!/bin/bash
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=synth_verify_20260325
#SBATCH --output=/gpfs1/home/w/t/wthomps3/CSDS/side_projects/heterogeneous-infection-kernels/logs/synth_verify_20260325_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=will.w.thompson@gmail.com

cd /gpfs1/home/w/t/wthomps3/CSDS/side_projects/heterogeneous-infection-kernels
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH

echo "Running on host: $(hostname)"
echo "Start time: $(date)"
echo "SYNTHETIC VERIFICATION -- SC, CC, NP EB K=1 on n=512 SBM -- March 25 2026"
echo "Goal: confirm NP EB recovers known NP kernel from full population M/N"

mkdir -p logs

uv run snakemake inference \
  --configfile configs/synth_verify_20260325.yaml \
  --profile workflow/profiles/local \
  --cores 4 \
  --rerun-incomplete

echo "End time: $(date)"
echo "Exit code: $?"
