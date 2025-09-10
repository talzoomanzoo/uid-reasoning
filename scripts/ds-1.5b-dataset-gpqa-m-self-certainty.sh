python ../src/run_direct_gen_uid_dev_viz.py --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
   --dataset_name gpqa \
   --split main \
   --self-certainty True \
   --batch_size 448 \
   --data_limit 448 \
   --sample_limit 5