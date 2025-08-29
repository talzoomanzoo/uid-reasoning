#!/bin/bash
set -x

export PYTHONPATH=$PYTHONPATH:$(pwd)/../
export WORLD_SIZE=4
export TRITON_CACHE_DIR="~/triton-cache"  # Avoid NFS warning
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

deepspeed --num_gpus 4 --module openrlhf.cli.train_sft \
   --max_len=32768 \
   --dataset=talzoomanzoo/lowest_uid_gini_sft_ds-1.5b \
   --input_key=Question \
   --output_key=text \
   --train_batch_size=8 \
   --micro_train_batch_size=2 \
   --pretrain=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
   --save_path=./checkpoint/deepseek-1.5b-lowest-uid-gini-ds-1.5b-fft \
   --packing_samples \
   --logging_steps=1 \
   --eval_steps=-1 \
   --zero_stage=2 \
   --max_epochs=10 \
   --learning_rate=5e-6 \
   --gradient_checkpointing \
   --bf16 \
   --use_wandb=True \
   --wandb_org=mjgwak \
   --wandb_project=uid_gini_sft\
   --wandb_run_name=deepseek-1.5b-lowest-uid-gini-ds-1.5b-fft-$(date +%m%d%H%M)