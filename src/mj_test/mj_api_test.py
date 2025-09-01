import os
import argparse
import asyncio
import json
import re
import base64
import time

from io import BytesIO
from copy import deepcopy
from tqdm.asyncio import tqdm_asyncio
from langchain_together import ChatTogether, Together
from datasets import load_dataset
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.callbacks import get_openai_callback


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="/scratch/mjgwak/uid-reasoning/data/AIME/train.json")
    parser.add_argument("--output_path", type=str, default="./outputs")
    parser.add_argument("--output_filename", type=str, default="aime_test")
    parser.add_argument("--prompt_path", type=str, default="./mj_prompt_test.json")
    parser.add_argument("--api_key", type=str, default=os.environ["TOGETHERAI_API_KEY"])
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", choices=["qwen-1.7b","qwen3-14b-base","deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "deepseek-r1-distill-qwen-1.5b"], help="Model to use for test")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of concurrent async tasks")
    parser.add_argument("--debug", action="store_true", help="Whether to run in debug mode")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index of the dataset")
    parser.add_argument("--end_idx", type=int, default=10, help="End index of the dataset")
    return parser.parse_args()


def generate_input(item):
    with open(args.prompt_path, "r") as f:
        prompt = json.load(f)
    instruction = prompt["instruction"]
    template = prompt["template"]

    # user prompt
    user_content = template.format(
        instruction=instruction,
        id=item["id"],
        year=item["year"],
        problem_number=item["problem_number"],
        question=item["Question"],
        answer=item["answer"]
    )

    messages = [HumanMessage(content=user_content)]
    return messages


async def process_item(instance, llm):
    # save original instance
    result = deepcopy(instance)

    if instance["answer"] == "":
        result["answer_invalid_format"] = True
    else:
        result["answer_invalid_format"] = False
    
    # generate
    final_input = generate_input(instance)
    response = await llm.agenerate([final_input])
    result['answer'] = response.generations[0][0].text
    if result['answer'] is None:
        result["answer_invalid_format"] = True
    else:
        result["answer_invalid_format"] = False
    return result["answer"]


async def process_dataset(dataset, llm, batch_size, args):
    results = []
    total_cost = 0
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(dataset)-1)//batch_size + 1} (size: {len(batch)})")
        with get_openai_callback() as cb:
            tasks = [process_item(instance, llm) for instance in batch]
            batch_results = await tqdm_asyncio.gather(*tasks, desc=f"Processing batch {i//batch_size + 1}")
            results.extend(batch_results)
            
            if args.model == "qwen-1.7b": # 0.15, 0.6
                batch_cost = 0
            elif args.model == "qwen3-14b-base":
                batch_cost = 0
            elif args.model == "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B":
                batch_cost = (cb.completion_tokens*0.0000016)*10
            elif args.model == "deepseek-r1-distill-qwen-1.5b":
                batch_cost = 0
            else:
                batch_cost = cb.total_cost
            
            total_cost += batch_cost
            print(f"Batch Cost: ${batch_cost:.6f}")
            print(f"Total Cost so far: ${total_cost:.6f}")
            print(f"Total tokens: {cb.total_tokens}")
            print(f"Prompt tokens: {cb.prompt_tokens}")
            print(f"Completion tokens: {cb.completion_tokens}")
    
    return results, total_cost


async def annotate_dataset_async(dataset, llm, batch_size, args):
    return await process_dataset(dataset, llm, batch_size, args)


async def annotate_dataset(args):
    with open(args.dataset, "r") as f:
        dataset = json.load(f)
    print(args.dataset)
    if args.debug:
        dataset = dataset[args.start_idx:args.end_idx]
    else:
        dataset = dataset[args.start_idx:args.end_idx]

    # load model
    llm = ChatTogether(model=args.model, api_key=args.api_key)

    # annotate dataset
    start_time = time.time()
    results, total_cost = await annotate_dataset_async(dataset, llm, args.batch_size, args)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution Time: {execution_time:.2f} seconds")
    print(f"Total Cost (USD): ${total_cost:.6f}")
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)

    if args.debug:
        output_filename = f"{args.output_filename}_{args.model}_{args.start_idx}_{args.end_idx}.json"
    else:
        output_filename = f"{args.output_filename}_{args.model}.json"
    with open(os.path.join(args.output_path, output_filename), "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(annotate_dataset(args))