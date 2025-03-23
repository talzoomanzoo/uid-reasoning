from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# === Define base and adapter ===
base_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
adapter_path = "talzoomanzoo/s1K-DeepSeek-R1-Distill-Qwen-1.5B-20250323_092434"

# === Load base model ===
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    trust_remote_code=True,
)

# === Load adapter and merge ===
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.merge_and_unload()

# === Save merged model ===
model.push_to_hub("talzoomanzoo/merged-s1K-DeepSeek-1.5B")

# === Save tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.push_to_hub("talzoomanzoo/merged-s1K-DeepSeek-1.5B")