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

# Run scripts sequentially
run_script "high_uid_variance_sft_ds.sh"
run_script "low_uid_variance_sft_ds.sh"

echo "All scripts completed successfully!"