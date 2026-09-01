#!/bin/bash
#SBATCH --job-name=vhh_pipeline_test
#SBATCH --output=logs/pipeline_test_%j.out
#SBATCH --error=logs/pipeline_test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --partition=gpu3090
#SBATCH --qos=gpudebug
#SBATCH --gres=gpu:1

# VHHCorpus-2M Real Data Pipeline Smoke Test
# Tests complete training pipeline with real TSV data

echo "=================================================="
echo "VHH Germline-Absorbing Diffusion Pipeline Test"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Activate conda environment (no module loading needed)
echo "Activating conda environment: sedd_vhh"
source ~/.bashrc
conda activate sedd_vhh

# Verify environment
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA version: $(python -c 'import torch; print(torch.version.cuda)')"
echo ""

# Set environment
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Create logs directory
mkdir -p logs

# Verify data file
DATA_FILE="/gpfs/work/bio/zhengtaoqi24/germline/VHHCorpus-2M_top1_pairs_clean.tsv"
echo "Data file: $DATA_FILE"
if [ -f "$DATA_FILE" ]; then
    echo "  [OK] File exists"
    echo "  Size: $(du -h $DATA_FILE | cut -f1)"
    echo "  Lines: $(wc -l < $DATA_FILE)"
else
    echo "  [FAIL] File not found!"
    exit 1
fi
echo ""

# Run smoke test
echo "=================================================="
echo "Running pipeline smoke test..."
echo "=================================================="
python test_real_germline_pipeline_supercomputer.py

EXIT_CODE=$?

echo ""
echo "=================================================="
echo "Test Complete"
echo "=================================================="
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "STATUS: SUCCESS [OK]"
    echo ""
    echo "Pipeline verified:"
    echo "  [OK] Real TSV loaded (~1.6M samples)"
    echo "  [OK] 20-state vocabulary"
    echo "  [OK] Token range [0, 19]"
    echo "  [OK] Forward/backward pass"
    echo "  [OK] Germline proj gradients"
    echo "  [OK] Validation"
    echo "  [OK] REAL reverse diffusion sampling"
    echo ""
    echo "Ready for full training!"
else
    echo "STATUS: FAILED [FAIL]"
    echo "Check error log for details."
fi

exit $EXIT_CODE
