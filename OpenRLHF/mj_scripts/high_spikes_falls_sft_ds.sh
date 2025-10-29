#!/bin/bash
set -x

export PYTHONPATH=$PYTHONPATH:$(pwd)/../
export WORLD_SIZE=4
export TRITON_CACHE_DIR="~/triton-cache"  # Avoid NFS warning
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

deepspeed --num_gpus 4 --module openrlhf.cli.train_sft \
   --max_len=32768 \
   --dataset=talzoomanzoo/DS_QWEN_HIGHEST_SPIKES_FALLS \
   --input_key=Question \
   --output_key=Output_highest \
   --train_batch_size=4 \
   --micro_train_batch_size=1 \
   --pretrain=Qwen/Qwen2.5-Math-1.5B \
   --save_path=./checkpoint/qwen2.5-math-1.5b-highest-spikes-falls-ds-1.5b-fft \
   --packing_samples \
   --flash_attn \
   --logging_steps=1 \
   --eval_steps=-1 \
   --zero_stage=2 \
   --max_epochs=10 \
   --learning_rate=4e-5 \
   --gradient_checkpointing \
   --bf16 \
   --use_wandb=True \
   --wandb_org=mjgwak \
   --wandb_project=spikes_falls_sft \
   --wandb_run_name=qwen2.5-math-1.5b-highest-spikes-falls-fft-$(date +%m%d%H%M)