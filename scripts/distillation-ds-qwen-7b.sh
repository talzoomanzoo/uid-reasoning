export VLLM_BATCH_INVARIANT=1

python ../src/run_direct_gen_uid_dev_viz.py --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
   --dataset_name aime \
   --split train \
   --batch_size 400 \
   --data_limit 400 \
   --sample_limit 5