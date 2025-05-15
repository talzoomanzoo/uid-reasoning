import asyncio
from concurrent.futures import ThreadPoolExecutor
from vllm import LLM, SamplingParams
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

# Initialize the LLM
llm = LLM(
    model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    tokenizer="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    dtype=torch.float16, 
    quantization="bitsandbytes",  
    load_format="bitsandbytes",  
    tensor_parallel_size=4,
    gpu_memory_utilization=0.7,  
    max_model_len=4096,  
    enforce_eager=True,
)

# Define the prompts and sampling parameters
prompts = ["Hello, my name is", "The capital of France is"]
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=50)

# Function to run the synchronous generate method
def sync_generate(prompts, sampling_params):
    return llm.generate(prompts, sampling_params=sampling_params)

# Asynchronous wrapper for the synchronous generate function
async def async_generate(prompts, sampling_params):
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, sync_generate, prompts, sampling_params)

# Main asynchronous function
async def main():
    outputs = await async_generate(prompts, sampling_params)
    for output in outputs:
        print(output.outputs[0].text)

# Run the main function
asyncio.run(main())
