python ../src/analysis_by_uid.py \
 --input /scratch/mjgwak/uid-reasoning/scripts/outputs/runs.baselines/aime.deepseek-r1-distill-qwen-7b.direct/train.10.28,8:49-5-thinksegFalse-step50.json \
 --input2 /scratch/mjgwak/uid-reasoning/scripts/outputs/runs.baselines/aime.deepseek-r1-distill-qwen-7b.direct/train.10.28,8:49-5-thinksegFalse-step50.metrics.json \
 --select_traces \
 --select_traces_by_spikes_falls \
 --outdir ../src/analysis_out_ds