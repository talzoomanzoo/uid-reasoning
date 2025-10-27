#!/bin/bash
set -x

export PYTHONPATH=$PYTHONPATH:$(pwd)/../
export WORLD_SIZE=4
export TRITON_CACHE_DIR="~/triton-cache"  # Avoid NFS warning
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

deepspeed --num_gpus 4 --module openrlhf.cli.train_sft \
   --max_len=32768 \
   --dataset=talzoomanzoo/QWEN3_HIGHEST_VARIANCE \
   --input_key=Question \
   --output_key=Output_highest \
   --train_batch_size=8 \
   --micro_train_batch_size=2 \
   --pretrain=Qwen/Qwen3-0.6B \
   --save_path=./checkpoint/qwen3-0.6b-highest-uid-variance-fft \
   --packing_samples \
   --flash_attn \
   --logging_steps=1 \
   --eval_steps=-1 \
   --zero_stage=2 \
   --max_epochs=3 \
   --learning_rate=5e-6 \
   --gradient_checkpointing \
   --bf16 \
   --use_wandb=True \
   --wandb_org=mjgwak \
   --wandb_project=uid_variance_sft \
   --wandb_run_name=qwen3-0.6b-highest-uid-variance-fft-$(date +%m%d%H%M)