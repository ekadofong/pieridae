#!/bin/bash
# BYOL Analysis for macOS with Apple Silicon GPU (MPS)
# Usage: ./run_byol_macos.sh [train|analyze|full] [additional_args...]

# Set strict error handling
set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default parameters
MODE="${1:-full}"
shift || true  # Remove first argument, ignore error if no args

# Set up environment variables
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$(sysctl -n hw.ncpu)

# Check if MPS (Metal Performance Shaders) is available
echo "Checking Metal Performance Shaders (MPS) availability..."
python3 -c "
import torch
if torch.backends.mps.is_available():
    print('✅ MPS (Apple Silicon GPU) is available')
    print(f'Device: {torch.backends.mps.is_built()}')
else:
    print('❌ MPS not available, will use CPU')
    if not torch.backends.mps.is_built():
        print('PyTorch was not compiled with MPS support')
"

# Set up paths
DATA_DIR="$PROJECT_ROOT/local_data"
OUTPUT_DIR="$PROJECT_ROOT/byol_results/macos_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Check if data exists
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Data directory not found: $DATA_DIR"
    echo "Please ensure your data is in the correct location"
    exit 1
fi

echo "=== BYOL macOS Analysis ==="
echo "Mode: $MODE"
echo "Project root: $PROJECT_ROOT"
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "CPU cores: $OMP_NUM_THREADS"
echo "Started: $(date)"
echo "=========================="

# Set up logging
LOG_FILE="$OUTPUT_DIR/byol_macos.log"

# Run analysis
echo "Starting BYOL analysis (macOS with MPS acceleration)..."
python3 "$SCRIPT_DIR/byol_cluster_analysis.py" \
    --mode "$MODE" \
    --data-path "$DATA_DIR/pieridae_output" \
    --output-path "$OUTPUT_DIR" \
    "$@" \
    2>&1 | tee "$LOG_FILE"

echo "=== Analysis Complete ==="
echo "Results saved to: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "Finished: $(date)"
echo "========================="

# Optional: Open results directory
if command -v open >/dev/null 2>&1; then
    echo "Opening results directory..."
    open "$OUTPUT_DIR"
fi