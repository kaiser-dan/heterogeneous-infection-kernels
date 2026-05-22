#!/bin/bash
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=twitter_k123_v2
#SBATCH --output=/gpfs1/home/w/t/wthomps3/CSDS/side_projects/heterogeneous-infection-kernels/logs/twitter_k123_v2_20260325_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=will.w.thompson@gmail.com

# Twitter survival full-pop -- SC, CC, NP at K=1,2,3 -- 4 chains/1000/1000
# V_max_local=1000, kmax=20. 6h wall for 9 fits with V_max sampling.
# March 26 2026

cd /gpfs1/home/w/t/wthomps3/CSDS/side_projects/heterogeneous-infection-kernels
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH

echo "Running on host: $(hostname)"
echo "Start time: $(date)"
echo "Twitter K=1,2,3 model selection v2 -- 9 fits -- March 26 2026"

mkdir -p logs

# Unlock in case a previous job left a stale lock
uv run snakemake --unlock --configfile configs/twitter_k123_v2_20260325.yaml --profile workflow/profiles/local 2>/dev/null || true

uv run snakemake inference \
  --configfile configs/twitter_k123_v2_20260325.yaml \
  --profile workflow/profiles/local \
  --cores 8 \
  --rerun-incomplete

echo "End time: $(date)"
echo "Exit code: $?"
