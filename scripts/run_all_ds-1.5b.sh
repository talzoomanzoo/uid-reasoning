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
run_script "ds-1.5b-aime.sh"
run_script "ds-1.5b-high-gini.sh"
run_script "ds-1.5b-high-shannon.sh"
run_script "ds-1.5b-high-variance.sh"
run_script "ds-1.5b-low-gini.sh"
run_script "ds-1.5b-low-shannon.sh"
run_script "ds-1.5b-low-variance.sh"

echo "All scripts completed successfully!"