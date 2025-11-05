python ../src/run_direct_gen_uid_dev_viz.py --model_path talzoomanzoo/qwen3-0.6b-lowest-spikes-falls-fft \
   --dataset_name hmmt \
   --split test \
   --batch_size 30 \
   --data_limit 30 \
   --sample_limit 5

python ../src/run_direct_gen_uid_dev_viz.py --model_path talzoomanzoo/qwen3-0.6b-highest-uid-variance-fft \
   --dataset_name aime \
   --split test \
   --batch_size 30 \
   --data_limit 30 \
   --sample_limit 5

python ../src/run_direct_gen_uid_dev_viz.py --model_path talzoomanzoo/qwen3-0.6b-highest-uid-variance-fft \
   --dataset_name hmmt \
   --split test \
   --batch_size 30 \
   --data_limit 30 \
   --sample_limit 5

python ../src/run_direct_gen_uid_dev_viz.py --model_path talzoomanzoo/qwen3-0.6b-lowest-uid-variance-fft \
   --dataset_name aime \
   --split test \
   --batch_size 30 \
   --data_limit 30 \
   --sample_limit 5

python ../src/run_direct_gen_uid_dev_viz.py --model_path talzoomanzoo/qwen3-0.6b-lowest-uid-variance-fft \
   --dataset_name hmmt \
   --split test \
   --batch_size 30 \
   --data_limit 30 \
   --sample_limit 5