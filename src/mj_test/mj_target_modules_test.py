from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        print(name)
