#!/bin/bash
#SBATCH --job-name=byol_full_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --partition=gpu-short
#SBATCH --account merian

# Set strict error handling
set -euo pipefail

echo "Job started on: $(date)"
echo "Running on node: $HOSTNAME"
echo "Job ID: $SLURM_JOB_ID"

# Load necessary modules
module purge
module load anaconda3/2023.9

# Activate conda environment
conda activate merenv

# Set environment variables
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export PYTHONPATH=$SLURM_SUBMIT_DIR:$PYTHONPATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Create output directory
OUTPUT_DIR="/scratch/gpfs/$USER/byol_results/full_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Set up data access
DATA_DIR="$SLURM_SUBMIT_DIR/local_data"
SCRATCH_DATA_DIR="/scratch/gpfs/$USER/byol_data"

if [ -d "$DATA_DIR" ] && [ ! -d "$SCRATCH_DATA_DIR" ]; then
    echo "Copying data to scratch for better I/O performance..."
    mkdir -p "$SCRATCH_DATA_DIR"
    rsync -av "$DATA_DIR/" "$SCRATCH_DATA_DIR/"
    echo "Data copy completed"
fi

# Check GPU
echo "GPU Information:"
nvidia-smi

# Run full BYOL pipeline
echo "Starting full BYOL pipeline (train + analyze)..."
LOG_FILE="$OUTPUT_DIR/slurm_job_${SLURM_JOB_ID}.log"

python3 "$SLURM_SUBMIT_DIR/scripts/byol/byol_cluster_analysis.py" \
    --config "$SLURM_SUBMIT_DIR/scripts/byol/byol_config.yaml" \
    --mode full \
    --data-path "$SCRATCH_DATA_DIR/pieridae_output" \
    --output-path "$OUTPUT_DIR" \
    2>&1 | tee "$LOG_FILE"

# Copy results back
FINAL_OUTPUT_DIR="$SLURM_SUBMIT_DIR/byol_results/full_$(basename $OUTPUT_DIR)"
mkdir -p "$FINAL_OUTPUT_DIR"
rsync -av "$OUTPUT_DIR/" "$FINAL_OUTPUT_DIR/"

echo "Full pipeline completed successfully!"
echo "Results saved to: $FINAL_OUTPUT_DIR"
echo "Job finished on: $(date)"