#quant-sft-gt-lora-base-merged
from huggingface_hub import create_repo, upload_folder
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
from awq import AutoAWQForCausalLM
from huggingface_hub import login

token="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"
login(token=token)


# Define base and adapter
base_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
adapter_path = "talzoomanzoo/DeepSeek-R1-Distill-Qwen-7B-awq-rft-on-y-lora"

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    token=token,
    trust_remote_code=True,
)

# Load adapter and merge
model = PeftModel.from_pretrained(base_model, adapter_path, token=token)
model = model.merge_and_unload()

# Save merged model locally
save_dir = "./0514-deepseek-7b-awq-rft-on-y-lora-merged"
model.save_pretrained(save_dir)

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, token = token)
tokenizer.save_pretrained(save_dir)

model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

# Upload merged model to Huggingface
repo_id = "talzoomanzoo/DeepSeek-R1-Distill-Qwen-7B-awq-rft-on-y-lora-merged"
create_repo(repo_id, exist_ok=True, private=False, token = token)

upload_folder(
    repo_id=repo_id,
    folder_path=save_dir,
    repo_type="model",
    token = token,
    commit_message="DeepSeek-R1-Distill-Qwen-7B-awq-rft-on-y-lora-merged"
)