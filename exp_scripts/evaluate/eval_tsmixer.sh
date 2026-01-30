#!/bin/bash

# Directory containing the YAML config files
MODEL_NAME="tsmixer"
NUM_PROCESSES=8
USE_GPU=1


CONFIG_DIR="configs/evaluate/$MODEL_NAME"

# Optional: Specify a pattern to match only your specific config files
# This pattern matches files like: 1152_576_S.yaml, 288_144_M.yaml, etc.
shopt -s nullglob  # Prevent literal '*_*_*.yaml' if no matches
config_files=("$CONFIG_DIR"/*_*_*.yaml)

# Check if directory exists
if [ ! -d "$CONFIG_DIR" ]; then
    echo "Error: Directory $CONFIG_DIR does not exist!"
    exit 1
fi

echo "Starting sequential execution of configuration files..."
echo "----------------------------------------"

# Loop through all matching YAML files
for config_file in "${config_files[@]}"; do
    # Skip if no files match the pattern
    [ -e "$config_file" ] || continue

    echo "Running: evaluate $MODEL_NAME --config_path $config_file"

    # Run the Python script with the current config
    quito-cli evaluate --config_path "$config_file" --num_processes $NUM_PROCESSES --use_gpu $USE_GPU

    # Check if the previous command succeeded
    if [ $? -ne 0 ]; then
        echo "Error: Failed to run $config_file"
        echo "Stopping execution."
        exit 1
    fi
    
    echo "Completed: $config_file"
    echo "----------------------------------------"
done

echo "All configurations executed successfully!"