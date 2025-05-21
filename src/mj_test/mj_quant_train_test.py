from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "talzoomanzoo/DeepSeek-R1-Distill-Qwen-7B-awq",
    trust_remote_code=True,
    torch_dtype="auto"
)

tokenizer = AutoTokenizer.from_pretrained(
    "talzoomanzoo/DeepSeek-R1-Distill-Qwen-7B-awq",
    trust_remote_code=True
)

print("Model and tokenizer loaded successfully.")
