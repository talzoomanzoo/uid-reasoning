python ../src/run_direct_gen_uid_dev_viz.py --model_path talzoomanzoo/deepseek-1.5b-lowest-spikes-falls-ds-1.5b-fft \
   --dataset_name hmmt \
   --split test \
   --batch_size 30 \
   --data_limit 30 \
   --sample_limit 5

python ../src/run_direct_gen_uid_dev_viz.py --model_path talzoomanzoo/deepseek-1.5b-highest-uid-variance-fft \
   --dataset_name aime \
   --split test \
   --batch_size 30 \
   --data_limit 30 \
   --sample_limit 5

python ../src/run_direct_gen_uid_dev_viz.py --model_path talzoomanzoo/deepseek-1.5b-highest-uid-variance-fft \
   --dataset_name hmmt \
   --split test \
   --batch_size 30 \
   --data_limit 30 \
   --sample_limit 5

python ../src/run_direct_gen_uid_dev_viz.py --model_path talzoomanzoo/deepseek-1.5b-lowest-uid-variance-fft \
   --dataset_name aime \
   --split test \
   --batch_size 30 \
   --data_limit 30 \
   --sample_limit 5

python ../src/run_direct_gen_uid_dev_viz.py --model_path talzoomanzoo/deepseek-1.5b-lowest-uid-variance-fft \
   --dataset_name hmmt \
   --split test \
   --batch_size 30 \
   --data_limit 30 \
   --sample_limit 5