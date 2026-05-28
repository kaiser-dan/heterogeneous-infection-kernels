#!/bin/bash
#SBATCH --job-name=oo_sweep_3-7-26
#SBATCH --partition=week
#SBATCH --array=0-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=results/oo_sweep_3-7-26/slurm_%A_%a.out

# ============================================================================
# OO Sim 165 Mixture Sweep: 12 tasks
#   Tasks 0-5:  Empirical Bayes (EB) priors  - 3 models x K=[1,2]
#   Tasks 6-11: Weak non-EB priors           - 3 models x K=[1,2]
# ============================================================================

cd ~/CSDS/side_projects/heterogeneous-infection-kernels
export PATH=$HOME/.cargo/bin:$PATH

echo "=== Task ${SLURM_ARRAY_TASK_ID} on $(hostname) ==="
echo "Start: $(date)"

# Tasks 0-5 -> EB config, tasks 6-11 -> WEAK config
# Within each block of 6, the sweep runner generates combos in order:
#   K=1,SC | K=1,CC | K=1,NP | K=2,SC | K=2,CC | K=2,NP
LOCAL_IDX=$((SLURM_ARRAY_TASK_ID % 6))

if [ $SLURM_ARRAY_TASK_ID -lt 6 ]; then
    CONFIG="configs/oo_sweep_3-7-26.yaml"
    REGIME="eb"
else
    CONFIG="configs/oo_sweep_3-7-26_weak.yaml"
    REGIME="weak"
fi

echo "Config: ${CONFIG} (regime=${REGIME}, local_idx=${LOCAL_IDX})"

# The sweep runner generates all 6 combos. We run only the one matching LOCAL_IDX.
# We use the sweep runner in non-HPC mode but only execute combo LOCAL_IDX.
# To do this, we call inference.py directly with the right instance params.

# Model mapping: combo index -> (K, stan_file)
# Product order: K=[1,2] x stan_file=[SC,CC,NP] -> 0:(1,SC) 1:(1,CC) 2:(1,NP) 3:(2,SC) 4:(2,CC) 5:(2,NP)
case $LOCAL_IDX in
    0) K=1; MODEL="src/models/mixture_simple_contagion.stan"; MODEL_SHORT="mixture_simple_contagion" ;;
    1) K=1; MODEL="src/models/mixture_complex_contagion.stan"; MODEL_SHORT="mixture_complex_contagion" ;;
    2) K=1; MODEL="src/models/mixture_non_parametric.stan"; MODEL_SHORT="mixture_non_parametric" ;;
    3) K=2; MODEL="src/models/mixture_simple_contagion.stan"; MODEL_SHORT="mixture_simple_contagion" ;;
    4) K=2; MODEL="src/models/mixture_complex_contagion.stan"; MODEL_SHORT="mixture_complex_contagion" ;;
    5) K=2; MODEL="src/models/mixture_non_parametric.stan"; MODEL_SHORT="mixture_non_parametric" ;;
esac

RUN_ID="${REGIME}_K${K}_${MODEL_SHORT}"
OUTPUT_DIR="results/oo_sweep_3-7-26/${RUN_ID}"
DATA_DIR="data/processed/oo/sim_165_backbone_disparity_kmax30"

mkdir -p "${OUTPUT_DIR}"

INSTANCE_PARAMS="{\"K\": ${K}, \"stan_file\": \"${MODEL}\"}"

echo "Run: ${RUN_ID}"
echo "Instance params: ${INSTANCE_PARAMS}"

uv run python -m src.models.inference \
    --config "${CONFIG}" \
    --exposure-summary "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --instance-params "${INSTANCE_PARAMS}"

echo "End: $(date)"
echo "Exit code: $?"
