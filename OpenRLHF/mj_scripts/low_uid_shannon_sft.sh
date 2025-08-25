#!/bin/bash
set -x

export PYTHONPATH=$PYTHONPATH:$(pwd)/../
export WORLD_SIZE=4
export TRITON_CACHE_DIR="/tmp/triton-cache"  # Avoid NFS warning
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

deepspeed --num_gpus 4 --module openrlhf.cli.train_sft \
   --ring_attn_size=4 \
   --ring_head_stride=1 \
   --max_len=32768 \
   --dataset=talzoomanzoo/lowest_uid_shannon_sft_ds-1.5b \
   --input_key=Question \
   --output_key=text \
   --train_batch_size=16 \
   --micro_train_batch_size=4 \
   --pretrain=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
   --save_path=./checkpoint/deepseek-1.5b-lowest-uid-shannon-ds-1.5b \
   --packing_samples \
   --logging_steps=1 \
   --eval_steps=-1 \
   --zero_stage=2 \
   --max_epochs=3 \
   --learning_rate=1e-5 \
   --gradient_checkpointing \
   --lora_rank=8 \
   --lora_alpha=8 \
   --lora_dropout=0.1 \
   --bf16 \
   --load_in_4bit \
   --use_wandb=True \
   --wandb_org=mjgwak \
   --target_modules q_proj \
   --target_modules k_proj \
   --target_modules v_proj \
   --target_modules o_proj \
   --target_modules gate_proj \
   --target_modules up_proj \
   --target_modules down_proj \
   --wandb_project=uid_shannon_sft\
   --wandb_run_name=deepseek-1.5b-lowest-uid-shannon-ds-1.5b-$(date +%m%d%H%M)