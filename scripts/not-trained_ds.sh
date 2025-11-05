python ../src/run_direct_gen_uid_dev_viz.py --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
   --dataset_name aime \
   --split test \
   --batch_size 30 \
   --data_limit 30 \
   --sample_limit 5

python ../src/run_direct_gen_uid_dev_viz.py --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
   --dataset_name hmmt \
   --split test \
   --batch_size 30 \
   --data_limit 30 \
   --sample_limit 5