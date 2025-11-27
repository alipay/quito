#!/bin/bash
################################################################################
# Run Dataset Quality Analysis in Background with nohup
#
# This script analyzes all item IDs across all indices (ind_1 to ind_5) in the
# open_hour_train dataset. It runs in the background and saves all output.
#
# Usage:
#   cd examples
#   bash run_quality_analysis.sh
#   
#   Or from project root:
#   bash examples/run_quality_analysis.sh
#
# The script will:
# - Run in background using nohup
# - Save console output to quality_analysis.log
# - Save results to quality_results.json
# - Support resume if interrupted
################################################################################

# Configuration
# Get the project root directory (parent of examples/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Paths relative to project root
DATA_DIR="examples/datasets/parquet_data/open_hour_train"
OUTPUT_FILE="$PROJECT_ROOT/examples/quality_results.json"
LOG_FILE="$PROJECT_ROOT/examples/quality_analysis.log"

# Analysis parameters
MAX_LENGTH=1000          # Truncate to last 1000 points per series (faster analysis, ~42 days of hourly data)
MAX_SERIES_PER_FILE=300  # 300 = sample 300 series per file (faster)
SAMPLING_STRATEGY="first" # Not used when MAX_SERIES_PER_FILE=0, but kept for consistency
PERIOD=24                # Hourly data with daily seasonality
USE_ALL_INDICES=true     # Analyze all ind_1 through ind_5

# Print configuration
echo "=================================="
echo "Dataset Quality Analysis"
echo "=================================="
echo "Data directory: $DATA_DIR"
echo "Output file: $OUTPUT_FILE"
echo "Log file: $LOG_FILE"
echo ""
echo "Settings:"
echo "  Max length per series: $MAX_LENGTH"
if [ "$MAX_SERIES_PER_FILE" -eq 0 ]; then
    echo "  Max series per file: ALL item IDs (no sampling)"
else
    echo "  Max series per file: $MAX_SERIES_PER_FILE (sampled)"
echo "  Sampling strategy: $SAMPLING_STRATEGY"
fi
echo "  Period: $PERIOD"
echo "  Use all indices: $USE_ALL_INDICES"
echo "  ADF stationarity test: ENABLED (requires 'arch' package)"
echo ""
if [ "$USE_ALL_INDICES" = true ]; then
echo "This will analyze ALL item IDs across ALL indices (ind_1 to ind_5)."
else
    echo "This will analyze ALL item IDs for the specified value column."
fi
echo "This may take several hours depending on data size."
echo ""

# Ask for confirmation
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Aborted."
    exit 1
fi

# Build command (run from project root)
cd "$PROJECT_ROOT"

# Use the python from 'llm' environment if available, otherwise default python
PYTHON_CMD="python"
if [ -f "/opt/miniconda3/envs/llm/bin/python" ]; then
    PYTHON_CMD="/opt/miniconda3/envs/llm/bin/python"
    echo "Using 'llm' conda environment python: $PYTHON_CMD"
fi

CMD="$PYTHON_CMD examples/analyze_open_hour_train_quality.py"
CMD="$CMD --data_dir $DATA_DIR"
CMD="$CMD --max_length $MAX_LENGTH"
CMD="$CMD --sampling_strategy $SAMPLING_STRATEGY"
CMD="$CMD --period $PERIOD"
CMD="$CMD --output_file $OUTPUT_FILE"

if [ "$USE_ALL_INDICES" = true ]; then
    CMD="$CMD --use_all_indices"
fi

# Always pass max_series_per_file (0 means no limit/all items)
    CMD="$CMD --max_series_per_file $MAX_SERIES_PER_FILE"

# ADF test enabled (stationarity testing - slower but more comprehensive)
# Note: Requires 'arch' package (pip install arch)
CMD="$CMD --compute_adf"

echo "Starting analysis in background..."
echo "Command: $CMD"
echo "Working directory: $PROJECT_ROOT"
echo ""

# Run with nohup
nohup $CMD > "$LOG_FILE" 2>&1 &

# Get the process ID
PID=$!

echo "=================================="
echo "Analysis started!"
echo "=================================="
echo "Process ID: $PID"
echo "Log file: $LOG_FILE"
echo "Output file: $OUTPUT_FILE"
echo ""
echo "To monitor progress:"
echo "  tail -f $LOG_FILE"
echo ""
echo "To check if still running:"
echo "  ps -p $PID"
echo ""
echo "To stop the analysis:"
echo "  kill $PID"
echo ""
echo "Results will be saved incrementally to $OUTPUT_FILE"
echo "You can resume if interrupted by running this script again."
echo "=================================="

# Save PID to file for easy reference
echo $PID > "$PROJECT_ROOT/examples/quality_analysis.pid"
echo "Process ID saved to examples/quality_analysis.pid"

