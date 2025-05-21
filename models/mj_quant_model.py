#quant-model-upload-to-hub
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from huggingface_hub import create_repo, upload_folder, login
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

token="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"
login(token=token)

model_path = 'talzoomanzoo/DeepSeek-R1-Distill-Qwen-7B-awq-rft-on-y-lora-merged'
quant_path = '/home/mjgwak/workspace/search-o1-dev/models/deepseek-7b-awq-rft-on-y-lora-merged'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# Load model
model = AutoAWQForCausalLM.from_pretrained(
    model_path, **{"low_cpu_mem_usage": True, "use_cache": False}
).cpu()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Set maximum sequence length
max_length = 16384  # Model's maximum context length
tokenizer.model_max_length = max_length

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

hub_repo_id = 'talzoomanzoo/rft-y-lora-y-true-merged-quant'

# (Optional) Create repo if it doesn't exist
create_repo(hub_repo_id, exist_ok=True)

# Upload to Hugging Face Hub
upload_folder(repo_id=hub_repo_id, folder_path=quant_path, path_in_repo=".")

print(f'Model is quantized and saved at "{quant_path}"')