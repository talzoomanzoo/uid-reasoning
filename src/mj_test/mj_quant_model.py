from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from huggingface_hub import HfApi, HfFolder, create_repo, upload_folder

model_path = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B'
quant_path = 'DeepSeek-R1-Distill-Qwen-14B-awq'  # local directory
hub_repo_id = 'talzoomanzoo/DeepSeek-R1-Distill-Qwen-14B-awq-8bit'  # on Hugging Face
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 8, "version": "GEMM" }

# Load and quantize
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model.quantize(tokenizer, quant_config=quant_config)

# Save locally
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

# (Optional) Create repo if it doesn't exist
create_repo(hub_repo_id, exist_ok=True)

# Upload to Hugging Face Hub
upload_folder(repo_id=hub_repo_id, folder_path=quant_path, path_in_repo=".")

print(f'Model pushed to Hugging Face Hub at https://huggingface.co/{hub_repo_id}')
