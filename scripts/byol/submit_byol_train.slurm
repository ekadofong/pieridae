#!/bin/bash
#SBATCH --job-name=byol_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=$USER@princeton.edu

# Set strict error handling
set -euo pipefail

echo "Job started on: $(date)"
echo "Running on node: $HOSTNAME"
echo "Job ID: $SLURM_JOB_ID"

# Load necessary modules
module purge
module load anaconda3/2023.9

# Activate conda environment
# Note: Replace 'merenv' with your actual environment name
conda activate merenv

# Set environment variables
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export PYTHONPATH=$SLURM_SUBMIT_DIR:$PYTHONPATH

# Create output directory on scratch (faster I/O)
OUTPUT_DIR="/scratch/gpfs/$USER/byol_results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Copy data to scratch if needed for better I/O performance
DATA_DIR="$SLURM_SUBMIT_DIR/local_data"
SCRATCH_DATA_DIR="/scratch/gpfs/$USER/byol_data"

if [ -d "$DATA_DIR" ] && [ ! -d "$SCRATCH_DATA_DIR" ]; then
    echo "Copying data to scratch for better I/O performance..."
    mkdir -p "$SCRATCH_DATA_DIR"
    rsync -av "$DATA_DIR/" "$SCRATCH_DATA_DIR/"
    echo "Data copy completed"
fi

# Check GPU availability
echo "GPU Information:"
nvidia-smi

# Set up logging
LOG_FILE="$OUTPUT_DIR/slurm_job_${SLURM_JOB_ID}.log"

# Run BYOL training
echo "Starting BYOL training..."
python3 "$SLURM_SUBMIT_DIR/scripts/byol/byol_cluster_analysis.py" \
    --mode train \
    --data-path "$SCRATCH_DATA_DIR/pieridae_output" \
    --output-path "$OUTPUT_DIR" \
    --epochs 50 \
    --batch-size 32 \
    --resume 2>&1 | tee "$LOG_FILE"

# Copy results back to submit directory if needed
FINAL_OUTPUT_DIR="$SLURM_SUBMIT_DIR/byol_results/$(basename $OUTPUT_DIR)"
mkdir -p "$FINAL_OUTPUT_DIR"
rsync -av "$OUTPUT_DIR/" "$FINAL_OUTPUT_DIR/"

echo "Training completed successfully!"
echo "Results saved to: $FINAL_OUTPUT_DIR"
echo "Job finished on: $(date)"

# Cleanup scratch data (optional, comment out if you want to keep it)
# rm -rf "$SCRATCH_DATA_DIR"