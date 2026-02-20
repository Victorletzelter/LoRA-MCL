#!/bin/bash

# Path to parent directory where all wandb folders are
BASE_DIR="./wandb"

# Find all wandb directories containing offline runs
find "$BASE_DIR" -type d -name 'offline-run-*' | while read run_dir; do
    echo "Syncing: $run_dir"
        wandb sync "$run_dir"
done

