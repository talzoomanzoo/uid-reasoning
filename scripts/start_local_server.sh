python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-1.7B \
    --port 8100 \
    --host 0.0.0.0 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9