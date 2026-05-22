#!/bin/bash
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=synth_model_sel_20260325
#SBATCH --output=/gpfs1/home/w/t/wthomps3/CSDS/side_projects/heterogeneous-infection-kernels/logs/synth_model_sel_20260325_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=will.w.thompson@gmail.com

# This script runs ONE K_true case. Pass --export K_TRUE=1 (or 2, or 3) when submitting.
# Or just pick the config from the array index.

cd /gpfs1/home/w/t/wthomps3/CSDS/side_projects/heterogeneous-infection-kernels
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH

# K_TRUE passed as env var from sbatch --export
if [ -z "$K_TRUE" ]; then
    echo "ERROR: K_TRUE env var not set. Use sbatch --export=K_TRUE=1"
    exit 1
fi

CONFIG="configs/synth_model_sel_k${K_TRUE}_20260325.yaml"

echo "Running on host: $(hostname)"
echo "Start time: $(date)"
echo "Synthetic model selection K_true=${K_TRUE} -- 27 fits, WAIC+BIC -- March 25 2026"
echo "Config: $CONFIG"

mkdir -p logs

uv run snakemake inference \
  --configfile "$CONFIG" \
  --profile workflow/profiles/local \
  --cores 4 \
  --rerun-incomplete

echo "End time: $(date)"
echo "Exit code: $?"
