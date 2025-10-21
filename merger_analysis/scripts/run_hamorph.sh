#!/bin/bash
#
# Wrapper script for H-alpha morphology analysis
#
# This script can be run directly for local testing or called by SLURM batch script.
# When SLURM_ARRAY_TASK_ID is set, it will process only the assigned chunk.
# Otherwise, it processes all galaxies in serial.
#
# Usage:
#   ./run_hamorph.sh [options passed to calc_hamorph.py]
#
# Examples:
#   # Local serial run
#   ./run_hamorph.sh --config ../configs/hamorph_config.yaml
#
#   # With specific H-alpha method
#   ./run_hamorph.sh --config ../configs/hamorph_config.yaml --ha-method zscale
#
#   # In SLURM job (chunk mode automatically enabled)
#   export SLURM_ARRAY_TASK_ID=0
#   ./run_hamorph.sh --config ../configs/hamorph_config.yaml

# Exit on error
set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default config path (relative to script location)
DEFAULT_CONFIG="${SCRIPT_DIR}/../configs/hamorph_config.yaml"

# Check if running in SLURM environment
if [ -n "${SLURM_ARRAY_TASK_ID}" ]; then
    echo "========================================="
    echo "H-alpha Morphology Analysis - SLURM Mode"
    echo "========================================="
    echo "Array task ID: ${SLURM_ARRAY_TASK_ID}"
    echo "Job ID: ${SLURM_JOB_ID:-N/A}"
    echo "Node: ${HOSTNAME}"
    echo "Started: $(date)"
    echo "========================================="
else
    echo "========================================="
    echo "H-alpha Morphology Analysis - Serial Mode"
    echo "========================================="
    echo "Started: $(date)"
    echo "========================================="
fi

# Parse arguments to find config file (for better error messages)
CONFIG_FILE=""
for arg in "$@"; do
    if [[ "$arg" == *.yaml ]] || [[ "$arg" == *.yml ]]; then
        CONFIG_FILE="$arg"
        break
    fi
done

# If no config specified in args, check if default exists
if [ -z "$CONFIG_FILE" ]; then
    if [ -f "$DEFAULT_CONFIG" ]; then
        CONFIG_FILE="$DEFAULT_CONFIG"
        echo "Using default config: $CONFIG_FILE"
    else
        echo "Warning: No config file found. Script may fail if config not specified in arguments."
    fi
else
    echo "Using config: $CONFIG_FILE"
fi

# Verify Python script exists
PYTHON_SCRIPT="${SCRIPT_DIR}/calc_hamorph.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found at $PYTHON_SCRIPT"
    exit 1
fi

# Print Python and environment info
echo ""
echo "Environment:"
echo "  Python: $(which python3)"
echo "  Python version: $(python3 --version 2>&1)"
echo "  Working directory: $(pwd)"
echo ""

# Run the analysis
echo "Running H-alpha morphology analysis..."
echo "Command: python3 $PYTHON_SCRIPT $@"
echo ""

# Execute with all arguments passed through
python3 "$PYTHON_SCRIPT" "$@"

EXIT_CODE=$?

echo ""
echo "========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Analysis completed successfully!"
else
    echo "Analysis failed with exit code: $EXIT_CODE"
fi
echo "Finished: $(date)"
echo "========================================="

exit $EXIT_CODE
