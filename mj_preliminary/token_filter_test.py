import argparse
import random
import sys
import os
from langchain_openai import ChatOpenAI
from transformers import AutoTokenizer
import torch

#code for manipulating <think> tokens

parser = argparse.ArgumentParser()
parser.add_argument("question", type=str)
parser.add_argument(
    "-m", "--model-name", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
)
parser.add_argument(
    "-r", "--replacements", nargs="+", default=["\nWait, but", "\nHmm", "\nSo"]
)
parser.add_argument("-t", "--min-thinking-tokens", type=int, default=128)
parser.add_argument("-p", "--prefill", default="")
parser.add_argument("--port", type=int, default=8100)
args = parser.parse_args()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set")

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
_, _start_think_token, end_think_token = tokenizer.encode("<think></think>")

# Use OpenAI-compatible vLLM server
model = ChatOpenAI(
    model=args.model_name,
    base_url=f"http://localhost:{args.port}/v1",
    temperature=0,
    max_tokens=3000,
    max_retries=100,
    openai_api_key=openai_api_key,
    streaming=True #check 

def reasoning_effort(question: str, min_thinking_tokens: int):
    """
    Generates a response ensuring a minimum amount of 'thinking' tokens.
    """
    input_text = f"User: {question}\nAssistant: <think>\n{args.prefill}"

    thinking_tokens = 0
    for chunk in model.stream(input_text):
        if thinking_tokens < min_thinking_tokens and ("</think>" in chunk.content or chunk.content.strip() == ""):
            replacement = random.choice(args.replacements)
            yield replacement
            thinking_tokens += len(tokenizer.encode(replacement))
        else:
            yield chunk
            thinking_tokens += len(tokenizer.encode(chunk.content))

for chunk in reasoning_effort(args.question, args.min_thinking_tokens):
    print(chunk, end="", flush=True)
