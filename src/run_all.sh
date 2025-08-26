#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Running highest-uid-gini..."
python run_direct_gen_uid_dev.py \
  --dataset_name math500 --split test \
  --model_path talzoomanzoo/deepseek-1.5b-highest-uid-gini-ds-1.5b-merged \
  --batch_size 500 --sample_limit 1 --data_limit 500

echo "Running lowest-uid-gini..."
python run_direct_gen_uid_dev.py \
  --dataset_name math500 --split test \
  --model_path talzoomanzoo/deepseek-1.5b-lowest-uid-gini-ds-1.5b-merged \
  --batch_size 500 --sample_limit 1 --data_limit 500

echo "Running highest-uid-variance..."
python run_direct_gen_uid_dev.py \
  --dataset_name math500 --split test \
  --model_path talzoomanzoo/deepseek-1.5b-highest-uid-variance-ds-1.5b-merged \
  --batch_size 500 --sample_limit 1 --data_limit 500

echo "Running lowest-uid-variance..."
python run_direct_gen_uid_dev.py \
  --dataset_name math500 --split test \
  --model_path talzoomanzoo/deepseek-1.5b-lowest-uid-variance-ds-1.5b-merged \
  --batch_size 500 --sample_limit 1 --data_limit 500

echo "All runs completed!"
