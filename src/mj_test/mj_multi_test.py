import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

# Initialize distributed
rank = int(os.environ["RANK"])
device = torch.device(f"cuda:{rank}")
torch.distributed.init_process_group("nccl", device_id=device)

# Retrieve tensor parallel model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    tp_plan="auto",
)

# Prepare input tokens
tokenizer = AutoTokenizer.from_pretrained(model_id)
prompt = "Can I help"
inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# Distributed run
# outputs = model(inputs)
# print(outputs)

outputs = model.generate(inputs, max_new_tokens=50)
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(output_text)

torch.distributed.destroy_process_group()
