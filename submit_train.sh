#!/bin/bash

#SBATCH --qos 4gpus
#SBATCH --partition gpu3090
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-12:00:00
#SBATCH --mem=20G
#SBATCH --mail-type=ALL
#SBATCH --job-name=GPU_TEST
#SBATCH --error=%j.err
#SBATCH --output=%j.out

# ============================================================
# Full training job for VHH Germline-Absorbing SEDD
# ============================================================

set -euo pipefail

PROJECT_DIR="/gpfs/work/bio/zhengtaoqi24/Score-Entropy-Discrete-Diffusion-main"
DATA_FILE="/gpfs/work/bio/zhengtaoqi24/germline/VHHCorpus-2M_top1_pairs_clean.tsv"
CONFIG_NAME="config_germline_vhh"

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# Activate the environment used by the verified local/smoke pipeline.
source ~/.bashrc
conda activate sedd_vhh

echo "============================================================"
echo "VHH Germline-Absorbing SEDD - FULL TRAINING"
echo "============================================================"
echo "Job ID:        ${SLURM_JOB_ID:-N/A}"
echo "Node:          ${SLURM_NODELIST:-N/A}"
echo "Start time:    $(date)"
echo "Project dir:   ${PROJECT_DIR}"
echo "Config:        ${CONFIG_NAME}"
echo "Data:          ${DATA_FILE}"
echo "Python:        $(python --version 2>&1)"
echo "PyTorch:       $(python -c 'import torch; print(torch.__version__)')"
echo "Torch CUDA:    $(python -c 'import torch; print(torch.version.cuda)')"
echo "CUDA available:$(python -c 'import torch; print(torch.cuda.is_available())')"
echo "============================================================"

# ---------- Pre-flight checks ----------
if [ ! -f "${DATA_FILE}" ]; then
    echo "[FAIL] Data file not found: ${DATA_FILE}"
    exit 1
fi

if [ ! -f "configs/${CONFIG_NAME}.yaml" ]; then
    echo "[FAIL] Config not found: configs/${CONFIG_NAME}.yaml"
    exit 1
fi

if [ ! -f "run_train.py" ]; then
    echo "[FAIL] run_train.py not found in ${PROJECT_DIR}"
    exit 1
fi

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("[FAIL] CUDA is not available inside the training job")
print("[OK] CUDA device:", torch.cuda.get_device_name(0))
PY

echo "[OK] Data file exists: $(du -h "${DATA_FILE}" | cut -f1)"
echo "[OK] Config exists: configs/${CONFIG_NAME}.yaml"
echo ""
echo "Starting full training..."
echo "Command: srun python -u run_train.py --config-name=${CONFIG_NAME} data.tsv_path=${DATA_FILE} ngpus=1"
echo "============================================================"

# ---------- Full training ----------
srun python -u run_train.py --config-name="${CONFIG_NAME}" data.tsv_path="${DATA_FILE}" ngpus=1

EXIT_CODE=$?

echo "============================================================"
echo "Training process finished"
echo "Exit code: ${EXIT_CODE}"
echo "End time: $(date)"
echo "============================================================"

exit ${EXIT_CODE}
