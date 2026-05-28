#!/bin/bash
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=oo_ado_direct
#SBATCH --output=/gpfs1/home/w/t/wthomps3/CSDS/side_projects/heterogeneous-infection-kernels/logs/oo_k123_adopters_direct_20260326_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=will.w.thompson@gmail.com

# OO adopters-only -- DIRECT inference.py calls, bypassing Snakemake DAG resolution
# which keeps using the wrong data_name (disparity_exposures instead of kmax30).
# Uses: data/summaries/oo/sim_165_backbone_disparity_kmax30.parquet + oo.adopters_only=True
# Outputs: results/oo_k123_adopters_20260326/oo/sim_165_backbone_disparity_kmax30/K~{k}/stan_file~{model}/
# March 26 2026

cd /gpfs1/home/w/t/wthomps3/CSDS/side_projects/heterogeneous-infection-kernels
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH

SUMMARY="data/summaries/oo/sim_165_backbone_disparity_kmax30.parquet"
CONFIG="configs/oo_k123_adopters_20260326.yaml"
SWEEP_NAME="oo_k123_adopters_20260326"
DATA_SOURCE="oo"
DATA_NAME="sim_165_backbone_disparity_kmax30"

echo "Running on host: $(hostname)"
echo "Start time: $(date)"
echo "OO adopters-only direct inference -- 9 fits -- March 26 2026"
echo "Summary: $SUMMARY"

mkdir -p logs

run_inference() {
    local k=$1
    local stan_file=$2
    local model_slug=$3
    local instance_id="K~${k}/stan_file~${model_slug}"
    local out_dir="results/${SWEEP_NAME}/${DATA_SOURCE}/${DATA_NAME}/${instance_id}"
    local params="{\"K\": ${k}, \"stan_file\": \"${stan_file}\"}"

    echo ""
    echo "=== K=${k} ${model_slug} ==="
    echo "Output: ${out_dir}"

    mkdir -p "${out_dir}"
    uv run python -m src.models.inference \
        --config "${CONFIG}" \
        --exposure-summary "${SUMMARY}" \
        --output-dir "${out_dir}" \
        --instance-params "${params}"
    echo "Exit code: $?"
}

# Run all 9 fits sequentially (8 cores each for Stan threading)
run_inference 1 "src/models/model_simple_contagion.stan"                    "model_simple_contagion"
run_inference 1 "src/models/model_complex_contagion.stan"                   "model_complex_contagion"
run_inference 1 "src/models/non_parametric_monotonic_empirical_bayes.stan"  "non_parametric_monotonic_empirical_bayes"
run_inference 2 "src/models/model_simple_contagion.stan"                    "model_simple_contagion"
run_inference 2 "src/models/model_complex_contagion.stan"                   "model_complex_contagion"
run_inference 2 "src/models/non_parametric_monotonic_empirical_bayes.stan"  "non_parametric_monotonic_empirical_bayes"
run_inference 3 "src/models/model_simple_contagion.stan"                    "model_simple_contagion"
run_inference 3 "src/models/model_complex_contagion.stan"                   "model_complex_contagion"
run_inference 3 "src/models/non_parametric_monotonic_empirical_bayes.stan"  "non_parametric_monotonic_empirical_bayes"

echo ""
echo "End time: $(date)"
echo "All 9 fits complete."
