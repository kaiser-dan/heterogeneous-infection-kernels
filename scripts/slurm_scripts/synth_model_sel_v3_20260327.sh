#!/bin/bash
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=synth_v3
#SBATCH --output=/gpfs1/home/w/t/wthomps3/CSDS/side_projects/heterogeneous-infection-kernels/logs/synth_model_sel_v3_20260327_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=will.w.thompson@gmail.com

# Synthetic model selection v3 -- K_true=1,2,3 all K_fit={1,2,3}
# 4 chains / 1000 warmup / 1000 samples for all 27 fits.
# Re-uses EXISTING datasets (synth_model_sel_k{1,2,3}_20260325) -- no re-simulation.
# Runs all 3 configs in parallel (each on 12 cores), 9 fits per config.
# March 27 2026

cd /gpfs1/home/w/t/wthomps3/CSDS/side_projects/heterogeneous-infection-kernels
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH

echo "Running on host: $(hostname)"
echo "Start time: $(date)"
echo "Synthetic v3 -- 27 fits total (3 K_true x 9 model-K combos)"

mkdir -p logs

run_config() {
    local config=$1
    local name=$2
    local cores=$3

    echo ""
    echo "=== Starting ${name} ==="
    uv run snakemake --unlock \
        --configfile "${config}" \
        --profile workflow/profiles/local 2>/dev/null || true

    uv run snakemake inference \
        --configfile "${config}" \
        --profile workflow/profiles/local \
        --cores "${cores}" \
        --rerun-incomplete
    echo "=== Done ${name} (exit $?) ==="
}

# Run all 3 configs in parallel, 12 cores each (4 chains * 3 concurrent fits)
run_config configs/synth_model_sel_k1_v3_20260327.yaml synth_k1_v3 12 &
PID1=$!

run_config configs/synth_model_sel_k2_v3_20260327.yaml synth_k2_v3 12 &
PID2=$!

run_config configs/synth_model_sel_k3_v3_20260327.yaml synth_k3_v3 12 &
PID3=$!

wait $PID1 && echo "k1 done" || echo "k1 FAILED"
wait $PID2 && echo "k2 done" || echo "k2 FAILED"
wait $PID3 && echo "k3 done" || echo "k3 FAILED"

echo ""
echo "End time: $(date)"
echo "All 27 fits complete."
