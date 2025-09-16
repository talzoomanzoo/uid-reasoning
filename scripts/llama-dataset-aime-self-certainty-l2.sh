python ../src/run_direct_gen_uid_dev_viz.py --model_path nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1 \
   --dataset_name aime \
   --split train \
   --self-certainty True \
   --batch_size 100 \
   --data_limit 100 \
   --sample_limit 5