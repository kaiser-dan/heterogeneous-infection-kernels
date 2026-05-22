#!/bin/bash
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=synth_ms_v2_20260325
#SBATCH --output=/gpfs1/home/w/t/wthomps3/CSDS/side_projects/heterogeneous-infection-kernels/logs/synth_ms_v2_20260325_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=will.w.thompson@gmail.com

# Reruns K_fit={2,3} for K_true={2,3} with 4 chains / 1000 warmup / 1000 samples.
# Fixes BIC (median_log_lik) and WAIC noise from the 1-chain v1 runs.
# Pass K_TRUE=2 or K_TRUE=3 via sbatch --export=K_TRUE=2

cd /gpfs1/home/w/t/wthomps3/CSDS/side_projects/heterogeneous-infection-kernels
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH

if [ -z "$K_TRUE" ]; then
    echo "ERROR: K_TRUE env var not set. Use sbatch --export=K_TRUE=2"
    exit 1
fi

CONFIG="configs/synth_model_sel_k${K_TRUE}_v2_20260325.yaml"

echo "Running on host: $(hostname)"
echo "Start time: $(date)"
echo "Synthetic model selection v2 K_true=${K_TRUE} -- K_fit=2,3 only, 4 chains/1000/1000 -- March 25 2026"
echo "Config: $CONFIG"

mkdir -p logs

uv run snakemake inference \
  --configfile "$CONFIG" \
  --profile workflow/profiles/local \
  --cores 8 \
  --rerun-incomplete

echo "End time: $(date)"
echo "Exit code: $?"
