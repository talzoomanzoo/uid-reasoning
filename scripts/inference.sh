for model_path in \
   Qwen/Qwen3-8B \
   deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
   deepseek-ai/DeepSeek-R1-Distill-Llama-8B
do
   for dataset_name in aime brumo hmmt
   do
      python ../src/run_inference.py --model_path "$model_path" \
         --dataset_name "$dataset_name" \
         --split test \
         --batch_size 30 \
         --data_limit 30 \
         --sample_limit 5
   done
done
