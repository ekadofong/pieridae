#!/bin/bash
#SBATCH --job-name=hamorph_analysis
#SBATCH --output=/scratch/gpfs/%u/hamorph_logs/hamorph_%A_%a.out
#SBATCH --error=/scratch/gpfs/%u/hamorph_logs/hamorph_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --partition=cpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=$USER@princeton.edu

################################################################################
# H-alpha Morphology Analysis - SLURM Job Array Submission Script
#
# This script runs H-alpha morphology measurements on Merian galaxies using
# SLURM job arrays for parallel processing. Each array task processes a chunk
# of the catalog.
#
# Usage:
#   # Submit all chunks (0-49, set in config)
#   sbatch --array=0-49 submit_hamorph.slurm
#
#   # Submit subset of chunks
#   sbatch --array=0-9 submit_hamorph.slurm
#
#   # Submit single chunk for testing
#   sbatch --array=0 submit_hamorph.slurm
#
#   # With custom config
#   sbatch --array=0-49 submit_hamorph.slurm /path/to/custom_config.yaml
#
# Notes:
#   - Number of chunks should match n_chunks in hamorph_config.yaml (default: 50)
#   - Adjust --array parameter to match n_chunks
#   - Each chunk processes ~1/50 of galaxies with available cutouts
#   - Logs saved to /scratch/gpfs/$USER/hamorph_logs/
#   - Results saved to output directory specified in config
#
################################################################################

# Set strict error handling
set -euo pipefail

# Print job information
echo "========================================================================"
echo "H-alpha Morphology Analysis - SLURM Job Array"
echo "========================================================================"
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Job Name:      ${SLURM_JOB_NAME}"
echo "Node:          ${HOSTNAME}"
echo "Started:       $(date)"
echo "User:          ${USER}"
echo "Submit Dir:    ${SLURM_SUBMIT_DIR}"
echo "========================================================================"

# Load necessary modules
echo "Loading modules..."
module purge
module load anaconda3/2023.9

# Activate conda environment
# Note: Adjust environment name if different
echo "Activating conda environment: merenv"
conda activate merenv

# Set environment variables
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Print environment info
echo ""
echo "Environment:"
echo "  Python:        $(which python3)"
echo "  Python version: $(python3 --version 2>&1)"
echo "  Conda env:     ${CONDA_DEFAULT_ENV}"
echo "  OMP threads:   ${OMP_NUM_THREADS}"
echo "  Working dir:   $(pwd)"
echo ""

# Create log directory if it doesn't exist
LOG_DIR="/scratch/gpfs/${USER}/hamorph_logs"
mkdir -p "${LOG_DIR}"
echo "Log directory: ${LOG_DIR}"

# Get config file from command line argument or use default
CONFIG_FILE="${1:-${SLURM_SUBMIT_DIR}/../configs/hamorph_config.yaml}"

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

echo "Config file: ${CONFIG_FILE}"

# Get output directory from config for logging purposes
OUTPUT_DIR=$(python3 -c "
import yaml
import sys
try:
    with open('${CONFIG_FILE}', 'r') as f:
        config = yaml.safe_load(f)
    print(config['output']['save_dir'])
except Exception as e:
    print('/scratch/gpfs/${USER}/hamorph_output', file=sys.stderr)
    sys.exit(0)
")

echo "Output directory: ${OUTPUT_DIR}"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Check available resources
echo ""
echo "Resources:"
echo "  CPUs:   ${SLURM_CPUS_PER_TASK}"
echo "  Memory: ${SLURM_MEM_PER_NODE}MB"
echo ""

# Print system info
echo "System information:"
free -h
echo ""

# Run the analysis using the wrapper script
WRAPPER_SCRIPT="${SLURM_SUBMIT_DIR}/run_hamorph.sh"

if [ ! -f "${WRAPPER_SCRIPT}" ]; then
    echo "ERROR: Wrapper script not found: ${WRAPPER_SCRIPT}"
    exit 1
fi

echo "========================================================================"
echo "Starting H-alpha morphology analysis (chunk ${SLURM_ARRAY_TASK_ID})..."
echo "========================================================================"
echo ""

# Execute the analysis
# The wrapper script will detect SLURM_ARRAY_TASK_ID automatically
bash "${WRAPPER_SCRIPT}" --config "${CONFIG_FILE}" --verbose

EXIT_CODE=$?

# Print summary
echo ""
echo "========================================================================"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "SUCCESS: Chunk ${SLURM_ARRAY_TASK_ID} completed successfully"
else
    echo "FAILURE: Chunk ${SLURM_ARRAY_TASK_ID} failed with exit code ${EXIT_CODE}"
fi
echo "Finished: $(date)"
echo "========================================================================"

# List output files created by this chunk
if [ ${EXIT_CODE} -eq 0 ] && [ -d "${OUTPUT_DIR}" ]; then
    echo ""
    echo "Output files created:"
    # Count .npy files created in last hour (rough estimate for this job)
    RECENT_FILES=$(find "${OUTPUT_DIR}" -name "*.npy" -type f -mmin -60 2>/dev/null | wc -l)
    echo "  Recent .npy files (last hour): ${RECENT_FILES}"
    echo ""
fi

exit ${EXIT_CODE}
