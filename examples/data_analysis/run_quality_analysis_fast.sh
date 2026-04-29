#!/bin/bash
################################################################################
# Run Dataset Quality Analysis (FAST MODE) in Background with nohup
#
# This is a faster version that samples a subset of item IDs for quicker results.
# Use this for exploratory analysis before running the full analysis.
#
# Usage:
#   cd examples
#   bash run_quality_analysis_fast.sh
#   
#   Or from project root:
#   bash examples/data_analysis/run_quality_analysis_fast.sh
################################################################################

# Configuration
# Get the project root directory (parent of examples/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Paths relative to project root
DATA_DIR="examples/datasets/parquet_data/open_hour_train"
OUTPUT_FILE="$PROJECT_ROOT/examples/quality_results_fast.json"
LOG_FILE="$PROJECT_ROOT/examples/quality_analysis_fast.log"

# Analysis parameters (FAST MODE)
MAX_LENGTH=2000          # Shorter: 2000 hours ≈ 83 days ≈ 2.7 months
MAX_SERIES_PER_FILE=50   # Sample only 50 item IDs per file
SAMPLING_STRATEGY="random" # Random sampling for representative sample
PERIOD=24                # Hourly data with daily seasonality
USE_ALL_INDICES=true     # Analyze all ind_1 through ind_5

# Print configuration
echo "=================================="
echo "Dataset Quality Analysis (FAST MODE)"
echo "=================================="
echo "Data directory: $DATA_DIR"
echo "Output file: $OUTPUT_FILE"
echo "Log file: $LOG_FILE"
echo ""
echo "Settings:"
echo "  Max length per series: $MAX_LENGTH"
echo "  Max series per file: $MAX_SERIES_PER_FILE (sampled)"
echo "  Sampling strategy: $SAMPLING_STRATEGY"
echo "  Period: $PERIOD"
echo "  Use all indices: $USE_ALL_INDICES"
echo ""
echo "This will sample $MAX_SERIES_PER_FILE item IDs per file across all indices."
echo "Expected runtime: 10-30 minutes (depending on data size)."
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
CMD="python examples/data_analysis/analyze_open_hour_train_quality.py"
CMD="$CMD --data_dir $DATA_DIR"
CMD="$CMD --max_length $MAX_LENGTH"
CMD="$CMD --max_series_per_file $MAX_SERIES_PER_FILE"
CMD="$CMD --sampling_strategy $SAMPLING_STRATEGY"
CMD="$CMD --period $PERIOD"
CMD="$CMD --output_file $OUTPUT_FILE"
CMD="$CMD --use_all_indices"

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
echo "Results will be saved to $OUTPUT_FILE"
echo "=================================="

# Save PID to file
echo $PID > "$PROJECT_ROOT/examples/quality_analysis_fast.pid"
echo "Process ID saved to examples/quality_analysis_fast.pid"

