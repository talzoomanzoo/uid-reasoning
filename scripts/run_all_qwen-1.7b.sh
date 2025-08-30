#!/bin/bash
set -e  # Exit on any error
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HUB_ENABLE_HF_TRANSFER=1
# Function to run script with logging
run_script() {
    local script_name=$1
    echo "=========================================="
    echo "Starting $script_name at $(date)"
    echo "=========================================="
    
    if bash "$script_name"; then
        echo "=========================================="
        echo "$script_name completed successfully at $(date)"
        echo "=========================================="
    else
        echo "=========================================="
        echo "ERROR: $script_name failed at $(date)"
        echo "=========================================="
        exit 1
    fi
}

# Run scripts sequentially
run_script "qwen-1.7b-aime.sh"
run_script "qwen-1.7b-aime-high-gini.sh"
run_script "qwen-1.7b-aime-high-shannon.sh"
run_script "qwen-1.7b-aime-high-variance.sh"
run_script "qwen-1.7b-aime-low-gini.sh"
run_script "qwen-1.7b-aime-low-shannon.sh"
run_script "qwen-1.7b-aime-low-variance.sh"

echo "All scripts completed successfully!"