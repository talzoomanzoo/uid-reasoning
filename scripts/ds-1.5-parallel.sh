#!/bin/bash

python ../src/run_direct_gen_uid_parallel.py \
    --dataset_name aime \
    --split test \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --sample_limit 1 \
    --batch_size 1 \
    --data_limit 1