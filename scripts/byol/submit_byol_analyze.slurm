#!/bin/bash
#SBATCH --job-name=byol_analyze
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --partition=cpu
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
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Set up directories
OUTPUT_DIR="/scratch/gpfs/$USER/byol_results/analysis_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Set data paths
DATA_DIR="$SLURM_SUBMIT_DIR/local_data"
SCRATCH_DATA_DIR="/scratch/gpfs/$USER/byol_data"

# Copy or link data if needed
if [ -d "$DATA_DIR" ] && [ ! -d "$SCRATCH_DATA_DIR" ]; then
    echo "Setting up data access on scratch..."
    mkdir -p "$SCRATCH_DATA_DIR"
    # Use symlinks for faster setup if on same filesystem
    if [ "$(stat -c %D "$DATA_DIR")" == "$(stat -c %D "$(dirname "$SCRATCH_DATA_DIR")")" ]; then
        ln -sf "$DATA_DIR"/* "$SCRATCH_DATA_DIR/"
        echo "Data symlinked to scratch"
    else
        rsync -av "$DATA_DIR/" "$SCRATCH_DATA_DIR/"
        echo "Data copied to scratch"
    fi
fi

# Check if trained model exists (from previous GPU job)
TRAINED_MODEL_DIR=""
# Look for the most recent training results
if [ -d "$SLURM_SUBMIT_DIR/byol_results" ]; then
    TRAINED_MODEL_DIR=$(find "$SLURM_SUBMIT_DIR/byol_results" -name "byol_final_model.pt" -type f -printf '%T@ %p\n' | sort -k1,1nr | head -1 | cut -d' ' -f2- | xargs dirname)
fi

if [ -z "$TRAINED_MODEL_DIR" ]; then
    echo "Warning: No trained model found. Will attempt to load from default location."
    TRAINED_MODEL_DIR="$SLURM_SUBMIT_DIR/byol_results"
fi

echo "Using trained model from: $TRAINED_MODEL_DIR"

# Set up logging
LOG_FILE="$OUTPUT_DIR/slurm_job_${SLURM_JOB_ID}.log"

# Check available memory and CPU cores
echo "System Information:"
echo "CPU cores: $SLURM_CPUS_PER_TASK"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "Available memory:"
free -h

# Run BYOL analysis (embedding extraction + dimensionality reduction)
echo "Starting BYOL analysis (PCA + UMAP + Similarity)..."
python3 "$SLURM_SUBMIT_DIR/scripts/byol/byol_cluster_analysis.py" \
    --mode analyze \
    --data-path "$SCRATCH_DATA_DIR/pieridae_output" \
    --output-path "$OUTPUT_DIR" \
    2>&1 | tee "$LOG_FILE"

# If we have a trained model from elsewhere, copy it to our output directory
if [ -f "$TRAINED_MODEL_DIR/byol_final_model.pt" ]; then
    cp "$TRAINED_MODEL_DIR/byol_final_model.pt" "$OUTPUT_DIR/"
    echo "Copied trained model to analysis output directory"
fi

# Copy embeddings and other artifacts if they exist
if [ -f "$TRAINED_MODEL_DIR/embeddings.npy" ]; then
    cp "$TRAINED_MODEL_DIR/embeddings.npy" "$OUTPUT_DIR/"
    echo "Copied embeddings to analysis output directory"
fi

# Copy results back to submit directory
FINAL_OUTPUT_DIR="$SLURM_SUBMIT_DIR/byol_results/analysis_$(basename $OUTPUT_DIR)"
mkdir -p "$FINAL_OUTPUT_DIR"
rsync -av "$OUTPUT_DIR/" "$FINAL_OUTPUT_DIR/"

echo "Analysis completed successfully!"
echo "Results saved to: $FINAL_OUTPUT_DIR"
echo "Job finished on: $(date)"

# Optional: Generate summary report
echo "=== Analysis Summary ===" | tee -a "$FINAL_OUTPUT_DIR/analysis_summary.txt"
if [ -f "$FINAL_OUTPUT_DIR/analysis_summary.json" ]; then
    python3 -c "
import json
with open('$FINAL_OUTPUT_DIR/analysis_summary.json', 'r') as f:
    summary = json.load(f)
print(f'Number of images processed: {summary.get(\"num_images\", \"N/A\")}')
print(f'Embedding dimension: {summary.get(\"embedding_dimension\", \"N/A\")}')
print(f'Completion time: {summary.get(\"completion_time\", \"N/A\")}')
" | tee -a "$FINAL_OUTPUT_DIR/analysis_summary.txt"
fi

# List output files
echo "Output files generated:" | tee -a "$FINAL_OUTPUT_DIR/analysis_summary.txt"
ls -la "$FINAL_OUTPUT_DIR" | tee -a "$FINAL_OUTPUT_DIR/analysis_summary.txt"