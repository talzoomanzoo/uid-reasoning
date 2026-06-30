cd "$(dirname "$0")"

found_output=0

for output_dir in ./outputs/runs.baselines/*.direct
do
   [ -d "$output_dir" ] || continue

   input_json=""
   for candidate in $(ls -t "${output_dir}"/test.*.json 2>/dev/null)
   do
      case "$candidate" in
         *.metrics.json) continue ;;
      esac

      metrics_json="${candidate%.json}.metrics.json"
      if [ -f "$metrics_json" ]; then
         input_json="$candidate"
         break
      fi
   done

   if [ -z "$input_json" ]; then
      echo "Skipping ${output_dir}: missing inference output or metrics"
      continue
   fi

   metrics_json="${input_json%.json}.metrics.json"
   found_output=1

   python ../src/analysis_by_uid.py \
      --input "$input_json" \
      --input2 "$metrics_json" \
      --outdir ./outputs/analysis_out
done

if [ "$found_output" -eq 0 ]; then
   echo "No complete inference outputs found under ./outputs/runs.baselines"
fi
