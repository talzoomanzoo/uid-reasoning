#!/bin/bash
set -e  # Exit on any error

# Pick a unique master port for each sub-run
pick_port() {
    echo $(shuf -i 20000-65000 -n 1)
}

# Function to run script with logging
run_script() {
    local script_name=$1
    export MASTER_ADDR=127.0.0.1
    export MASTER_PORT=$(pick_port)

    echo "=========================================="
    echo "Starting $script_name at $(date)"
    echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
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
run_script "high_spikes_falls_sft_ds.sh"
run_script "low_spikes_falls_sft_ds.sh"
run_script "high_uid_variance_sft_ds.sh"
run_script "low_uid_variance_sft_ds.sh"

echo "All scripts completed successfully!"
