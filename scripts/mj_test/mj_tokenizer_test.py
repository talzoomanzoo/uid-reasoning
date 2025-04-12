from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", trust_remote_code=True)
import pdb; pdb.set_trace()
print(tokenizer.add_special_tokens)
print(tokenizer.special_tokens_map)
print(tokenizer.special_tokens_map_extended)
