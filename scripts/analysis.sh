cd "$(dirname "$0")"

for model_short_name in \
   qwen3-8b \
   deepseek-r1-distill-qwen-7b \
   deepseek-r1-distill-llama-8b
do
   for dataset_name in aime brumo hmmt
   do
      output_dir="./outputs/runs.baselines/${dataset_name}.${model_short_name}.direct"
      input_json=$(ls -t "${output_dir}"/test.*-5.json 2>/dev/null | head -n 1)
      metrics_json="${input_json%.json}.metrics.json"

      if [ -z "$input_json" ] || [ ! -f "$metrics_json" ]; then
         echo "Skipping ${dataset_name}.${model_short_name}: missing inference output or metrics"
         continue
      fi

      python ../src/analysis_by_uid.py \
         --input "$input_json" \
         --input2 "$metrics_json" \
         --outdir ../src/analysis_out
   done
done
