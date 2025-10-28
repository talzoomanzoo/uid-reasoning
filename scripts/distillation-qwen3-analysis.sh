python ../src/analysis_by_uid.py \
 --input /scratch/mjgwak/uid-reasoning/scripts/outputs/runs.baselines/aime.qwen3-8b.direct/train.10.28,15:24-5-thinksegFalse-step50.json\
 --input2 /scratch/mjgwak/uid-reasoning/scripts/outputs/runs.baselines/aime.qwen3-8b.direct/train.10.28,15:24-5-thinksegFalse-step50.metrics.json \
 --select_traces \
 --select_traces_by_spikes_falls \
 --outdir ../src/analysis_out_qwen