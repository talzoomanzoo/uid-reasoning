python ../src/run_direct_gen_uid_dev_viz.py --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
   --dataset_name gsm8k \
   --split train \
   --self-certainty True \
   --batch_size 1000 \
   --data_limit 1000 \
   --sample_limit 5