#!/bin/bash
set -x

export PYTHONPATH=$PYTHONPATH:$(pwd)/../
export WORLD_SIZE=4
export TRITON_CACHE_DIR="~/triton-cache"  # Avoid NFS warning
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

deepspeed --num_gpus 4 --module openrlhf.cli.train_sft \
   --ring_attn_size=4 \
   --ring_head_stride=1 \
   --max_len=32768 \
   --dataset=talzoomanzoo/lowest_uid_gini_sft_qwen \
   --input_key=Question \
   --output_key=text \
   --train_batch_size=8 \
   --micro_train_batch_size=2 \
   --pretrain=Qwen/Qwen3-1.7B \
   --save_path=./checkpoint/deepseek-1.5b-lowest-uid-gini-qwen-1.7b \
   --packing_samples \
   --flash_attn \
   --logging_steps=1 \
   --eval_steps=-1 \
   --zero_stage=2 \
   --max_epochs=10 \
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
   --wandb_project=uid_gini_sft\
   --wandb_run_name=deepseek-1.5b-lowest-uid-gini-qwen-1.7b-$(date +%m%d%H%M)