#!/bin/bash
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=tw_k123_adopters
#SBATCH --output=/gpfs1/home/w/t/wthomps3/CSDS/side_projects/heterogeneous-infection-kernels/logs/twitter_k123_adopters_20260326_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=will.w.thompson@gmail.com

# Twitter survival -- adopters-only -- SC, CC, NP at K=1,2,3 -- 4 chains/1000/1000
# ~197 adopters from the cascade data. Much smaller than full-pop; no OOM risk.
# V_max_local=1000 applies but actual V << 1000, so all adopters included.
# March 26 2026

cd /gpfs1/home/w/t/wthomps3/CSDS/side_projects/heterogeneous-infection-kernels
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH

echo "Running on host: $(hostname)"
echo "Start time: $(date)"
echo "Twitter K=1,2,3 adopters-only -- 9 fits -- March 26 2026"

mkdir -p logs

uv run snakemake --unlock --configfile configs/twitter_k123_adopters_20260326.yaml --profile workflow/profiles/local 2>/dev/null || true

uv run snakemake inference \
  --configfile configs/twitter_k123_adopters_20260326.yaml \
  --profile workflow/profiles/local \
  --cores 8 \
  --rerun-incomplete

echo "End time: $(date)"
echo "Exit code: $?"
