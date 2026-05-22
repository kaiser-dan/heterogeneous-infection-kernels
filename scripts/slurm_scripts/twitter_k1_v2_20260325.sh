#!/bin/bash
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=tw_k1_v2
#SBATCH --output=/gpfs1/home/w/t/wthomps3/CSDS/side_projects/heterogeneous-infection-kernels/logs/twitter_k1_v2_20260325_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=will.w.thompson@gmail.com

# Twitter survival K=1 -- SC, CC, NP -- 4 chains/1000/1000
# Uses --nolock so K=1, K=2, K=3 jobs can run simultaneously.
# March 26 2026

cd /gpfs1/home/w/t/wthomps3/CSDS/side_projects/heterogeneous-infection-kernels
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH

echo "Running on host: $(hostname)"
echo "Start time: $(date)"
echo "Twitter K=1 model selection v2 -- March 26 2026"

mkdir -p logs

uv run snakemake inference \
  --configfile configs/twitter_k1_v2_20260325.yaml \
  --profile workflow/profiles/local \
  --cores 8 \
  --nolock \
  --rerun-incomplete

echo "End time: $(date)"
echo "Exit code: $?"
