#!/bin/bash
#SBATCH --partition=nvgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --job-name=tw_np_k3_r2
#SBATCH --output=/gpfs1/home/w/t/wthomps3/CSDS/side_projects/heterogeneous-infection-kernels/logs/twitter_np_k3_resubmit_20260326_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=will.w.thompson@gmail.com

# Twitter NP K=3 resubmit -- OOM fix
# Original job 3458797 killed with 3 oom_kill events at 5h43m on general (32G).
# SC K=3 and CC K=3 already complete in twitter_k123_v2_20260325/.
# This runs ONLY NP K=3 with 512G on nvgpu h2nodes (1TB RAM each).
# March 26 2026

cd /gpfs1/home/w/t/wthomps3/CSDS/side_projects/heterogeneous-infection-kernels
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH

echo "Running on host: $(hostname)"
echo "Start time: $(date)"
echo "Twitter NP K=3 resubmit -- nvgpu 512G -- March 26 2026"

mkdir -p logs

uv run snakemake inference \
  --configfile configs/twitter_np_k3_resubmit_20260326.yaml \
  --profile workflow/profiles/local \
  --cores 8 \
  --nolock \
  --rerun-incomplete

echo "End time: $(date)"
echo "Exit code: $?"
