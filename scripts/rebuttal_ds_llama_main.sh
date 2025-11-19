# Qwen/Qwen3-8B: seeds 42, 1234, 2025 for hmmt and brumo, with sample limits 5, 10, 15, 20
for seed in 42 1234 2025; do
  for dataset in aime hmmt brumo minervamath gpqa; do
    for sample_limit in 5; do
      # Set batch_size and data_limit based on dataset
      if [ "$dataset" = "minervamath" ]; then
        batch_size=272
        data_limit=272
      elif [ "$dataset" = "gpqa" ]; then
        batch_size=198
        data_limit=198
      else
        batch_size=30
        data_limit=30
      fi
      
      echo "Running deepseek-ai/DeepSeek-R1-Distill-Llama-8B with seed $seed on dataset $dataset with sample_limit $sample_limit"
      python ../src/run_direct_gen_uid_dev_viz.py --model_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
        --dataset_name $dataset \
        --seed $seed \
        --split test \
        --batch_size $batch_size \
        --data_limit $data_limit \
        --sample_limit $sample_limit
    done
  done
done