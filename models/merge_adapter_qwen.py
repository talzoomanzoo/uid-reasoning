from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Math-1.5B",
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(
    base_model,
    "talzoomanzoo/s1K-1.1-Qwen2.5-Math-1.5B-20250323_072421",
)

model = model.merge_and_unload()  # Apply LoRA weights into base

model.push_to_hub("talzoomanzoo/merged-s1K-Qwen2.5-Math-1.5B")

# Also save tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B", trust_remote_code=True)
tokenizer.push_to_hub("talzoomanzoo/merged-s1K-Qwen2.5-Math-1.5B")
