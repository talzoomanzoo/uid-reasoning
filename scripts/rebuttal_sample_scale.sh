#!/bin/bash

# Qwen/Qwen3-8B: seeds 42, 1234, 2025 for hmmt and brumo, with sample limits 5, 10, 15, 20
for seed in 42 1234 2025; do
  for dataset in hmmt brumo; do
    for sample_limit in 5 10 15 20; do
      echo "Running Qwen/Qwen3-8B with seed $seed on dataset $dataset with sample_limit $sample_limit"
      python ../src/run_direct_gen_uid_dev_viz.py --model_path Qwen/Qwen3-8B \
        --dataset_name $dataset \
        --seed $seed \
        --split test \
        --batch_size 30 \
        --data_limit 30 \
        --sample_limit $sample_limit
    done
  done
done

# deepseek-ai/DeepSeek-R1-Distill-Qwen-7B: seeds 42, 1234, 2025 for aime, hmmt, and brumo, with sample limits 5, 10, 15, 20
for seed in 42 1234 2025; do
  for dataset in aime hmmt brumo; do
    for sample_limit in 5 10 15 20; do
      echo "Running deepseek-ai/DeepSeek-R1-Distill-Qwen-7B with seed $seed on dataset $dataset with sample_limit $sample_limit"
      python ../src/run_direct_gen_uid_dev_viz.py --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        --dataset_name $dataset \
        --seed $seed \
        --split test \
        --batch_size 30 \
        --data_limit 30 \
        --sample_limit $sample_limit
    done
  done
done
