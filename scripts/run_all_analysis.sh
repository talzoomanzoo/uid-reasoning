#!/bin/bash
set -e  # Exit on any error

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
run_script "ds-1.5b-analysis.sh"
run_script "ds-7b-analysis.sh"
run_script "qwen3-1.7b-analysis.sh"
run_script "qwen3-4b-analysis.sh"

echo "All analysis scripts completed successfully"