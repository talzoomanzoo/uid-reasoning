import csv
import json
import random
import torch
import re
import os, time
import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from evaluate_uid_dev_viz import run_evaluation
from prompts import (
    get_task_instruction_math, 
    get_task_instruction_multi_choice, 
)
from tqdm import tqdm
import argparse
import asyncio

def format_question_with_choices(item):
    question = item['Question']
    choices = item.get('Choices')
    if not choices:
        return question

    choice_lines = []
    if isinstance(choices, dict):
        for label, text in choices.items():
            choice_lines.append(f"{label}. {text}")
    else:
        for idx, choice in enumerate(choices):
            label = chr(ord('A') + idx)
            text = choice[1] if isinstance(choice, (list, tuple)) and len(choice) > 1 else choice
            choice_lines.append(f"{label}. {text}")

    return f"{question}\n\nChoices:\n" + "\n".join(choice_lines)

def parse_args():
    parser = argparse.ArgumentParser(description="Run direct generation for various datasets and models.")

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )
    
    parser.add_argument(
        '--dataset_name', 
        type=str,
        required=True, 
        choices=['aime', 'brumo', 'hmmt', 'minervamath', 'gpqa', 'lsat_ar', 'lsat_lr'],
        help="Name of the dataset to use."
    )
    
    parser.add_argument(
        '--split',
        type=str, 
        required=True, 
        choices=['diamond', 'main', 'extended', 'train', 'test'],
        help="Dataset split to use."
    )
    
    parser.add_argument(
        '--subset_num', 
        type=int, 
        default=-1, 
        help="Number of examples to process. Defaults to all if not specified."
    )
    
    parser.add_argument(
        '--model_path', 
        type=str, 
        required=True,
        help="Path to the pre-trained model."
    )
    
    parser.add_argument(
        '--temperature', 
        type=float, 
        default=0.6, #default LRM temp 0.5~0.7 preferred, with 0.6 recommended
        help="Sampling temperature."
    )
    
    parser.add_argument(
        '--top_p', 
        type=float, 
        default=0.95, 
        help="Top-p sampling parameter."
    )
    
    parser.add_argument(
        '--top_k', 
        type=int, 
        default=20, 
        help="Top-k sampling parameter."
    )
    
    parser.add_argument(
        '--repetition_penalty', 
        type=float, 
        default=None, 
        help="Repetition penalty. If not set, defaults basplit sed on the model."
    )
    
    parser.add_argument(
        '--max_tokens', 
        type=int, 
        default=32768, 
        help="Maximum number of tokens to generate. If not set, defaults based on the model and dataset."
    )

    parser.add_argument(
        '--port',
        type=int,
        default=8100,
        help="Port to use for the OpenAI API."
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help="Batch size for the OpenAI API."
    )

    parser.add_argument(
        '--data_limit',
        type=int,
        default=-1,
        help="Number of examples to process. Defaults to all if not specified."
    )
    parser.add_argument(
        '--sample_limit',
        type=int,
        default=10,
        help="Number of examples to sample. Defaults to 10 if not specified."
    )

    parser.add_argument(
        '--skip_special_tokens',
        type=bool,
        default=False,
        help="Whether to skip special tokens. Defaults to False if not specified."
    )

    parser.add_argument(
        '--use_beam_search',
        type=bool,
        default=False,
        help="Whether to use beam search. Defaults to False if not specified."
    )
    
    parser.add_argument(
        '--self-certainty',
        type=bool,
        default=True,
        help="Whether to use self-certainty. Defaults to True if not specified."
    )

    parser.add_argument(
        '--confidence',
        type=bool,
        default=False,
        help="Whether to use confidence. Defaults to False if not specified."
    )
    
    parser.add_argument(
        '--entropy',
        type=bool,
        default=False,
        help="Whether to use entropy. Defaults to False if not specified."
    )
    
    return parser.parse_args()

async def main(args):
    dataset_name = args.dataset_name
    split = args.split
    subset_num = args.subset_num
    model_path = args.model_path
    temperature = args.temperature
    top_p = args.top_p
    top_k = args.top_k
    repetition_penalty = args.repetition_penalty
    max_tokens = args.max_tokens
    batch_size = args.batch_size
    data_limit = args.data_limit
    sample_limit = args.sample_limit
    skip_special_tokens = args.skip_special_tokens
    use_beam_search = args.use_beam_search
    self_certainty = args.self_certainty
    confidence = args.confidence
    entropy = args.entropy

    # Paths to supported datasets
    if dataset_name == 'aime':
        data_path = f'../data/AIME/{split}.json'
    elif dataset_name == 'brumo':
        data_path = f'../data/BRUMO/{split}.json'
    elif dataset_name == 'hmmt':
        data_path = f'../data/HMMT/{split}.json'
    elif dataset_name == 'minervamath':
        data_path = f'../data/MINERVA_MATH/{split}.json'
    elif dataset_name == 'gpqa':
        if split != 'diamond':
            raise ValueError("Only the GPQA diamond split is supported. Use --split diamond.")
        data_path = '../data/GPQA_DIAMOND_RAW/test.json'
    elif dataset_name == 'lsat_ar':
        data_path = f'../data/LSAT_AR/{split}.json'
    elif dataset_name == 'lsat_lr':
        data_path = f'../data/LSAT_LR/{split}.json'
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")
    
    # Load the model
    if "free" in model_path.lower():
        pass
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
    
    if 'qwen-14b' in model_path.lower():
        model_short_name = 'ds-qwen-14b'
    elif 'qwen3-4b' in model_path.lower():
        model_short_name = 'qwen3-4b'
    elif 'qwen3-1.7b' in model_path.lower():
        model_short_name = 'qwen3-1.7b'
    elif model_path.lower() == 'deepseek-r1-distill-qwen-7b-awq':
        model_short_name = 'ds-qwen-7b-awq'
    else:
        model_short_name = model_path.split('/')[-1].lower().replace('-instruct', '')

    # if model_short_name in ['qwq', 'ds-qwen-14b', 'ds-qwen-7b', 'ds-qwen-1.5b', 'sky-t1']:
    if model_short_name in ['emdr2']:
        if dataset_name in ['aime', 'brumo', 'hmmt', 'minervamath', 'gpqa', 'lsat_ar', 'lsat_lr']:
            output_dir = f'./outputs/{dataset_name}.{model_short_name}.direct'
        else:
            output_dir = f'./outputs/runs.qa/{dataset_name}.{model_short_name}.direct'
    else:
        output_dir = f'./outputs/runs.baselines/{dataset_name}.{model_short_name}.direct'
    os.makedirs(output_dir, exist_ok=True)

    llm = LLM(
                model=model_path,
                gpu_memory_utilization=0.90,
                max_model_len=32768,
                max_num_seqs=4,
                enforce_eager=False,
                dtype="bfloat16",
                tensor_parallel_size=4,
                swap_space=32,
                trust_remote_code=True,
        )
                
    sampling_params = SamplingParams(
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                max_tokens=max_tokens,
                skip_special_tokens=skip_special_tokens,
                include_stop_str_in_output=True,
                logprobs=1
        )


    # Load data
    with open(data_path, mode='r', encoding='utf-8') as json_file:
        filtered_data = json.load(json_file)
        filtered_data = filtered_data[:data_limit]
    
    # prepare input
    input_list = []
    for item in filtered_data:
        question = item['Question']
        if dataset_name in ['aime', 'brumo', 'hmmt', 'minervamath']:
            if 'qwq' in model_path.lower() or 'deepseek' in model_path.lower() or 'sky-t1' in model_path.lower() or 's1' in model_path.lower():
                user_prompt = get_task_instruction_math(question, model_name='qwq')
            else:
                user_prompt = get_task_instruction_math(question)

        elif dataset_name in ['gpqa', 'lsat_ar', 'lsat_lr']:
            question = format_question_with_choices(item)
            if 'qwq' in model_path.lower() or 'deepseek' in model_path.lower() or 'sky-t1' in model_path.lower():
                user_prompt = get_task_instruction_multi_choice(question, model_name='qwq')
            elif 'llama' in model_path.lower():
                user_prompt = get_task_instruction_multi_choice(question, model_name='llama')
            else:
                user_prompt = get_task_instruction_multi_choice(question)
        else:
            raise ValueError(f"Unsupported dataset_name: {dataset_name}")

        if "free" in model_path.lower():
            prompt = [{"role": "user", "content": user_prompt}]
        else:
            prompt = [{"role": "user", "content": user_prompt}]

        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        input_list.append(prompt)
    
    if subset_num != -1:
        input_list = input_list[:subset_num]
        filtered_data = filtered_data[:subset_num]
    
    # Set default max_tokens if not provided
    if max_tokens is None:
        if 'qwq' in model_path.lower() or 'deepseek' in model_path.lower() or 'sky-t1' in model_path.lower():
            if dataset_name in ['aime', 'brumo', 'hmmt', 'minervamath', 'gpqa', 'lsat_ar', 'lsat_lr']:
                max_tokens = 32768  
            else:
                max_tokens = 32768 
        else:
            max_tokens = 32768

    # Adjust max_tokens to fit within the model's context length
    max_tokens = min(max_tokens, 32768 - 243)  # Ensure total tokens do not exceed 32768

    async def generate_outputs(llm, input_batches, model_path, max_tokens, sample_limit, temperature, top_p, batch_size):
        output_list = []
        if len(input_batches) > batch_size:
            for start_index in tqdm(range(0, len(input_batches), batch_size)):
                cur_input_list = input_batches[start_index:start_index + batch_size]

                # Generate sample_limit outputs for each input in the batch
                for _ in range(sample_limit):  # Generate sample_limit times for each input
                    batch_output = await (asyncio.to_thread(llm.generate, cur_input_list, sampling_params=sampling_params))
                    output_list.extend(batch_output)
        else:

            # Generate sample_limit outputs for each input
            for _ in tqdm(range(sample_limit)):  # Generate sample_limit times for each input
                batch_output = await (asyncio.to_thread(llm.generate, input_batches, sampling_params=sampling_params))
                output_list.extend(batch_output)
        return output_list

    t_start = time.time()
    output_list = await generate_outputs(llm, input_list, model_path, max_tokens, sample_limit, temperature, top_p, batch_size)
    total_time = time.time() - t_start

    run_evaluation(
            filtered_data, 
            input_list, 
            output_list, 
            dataset_name, 
            output_dir, 
            total_time, 
            split,
            data_limit,
            sample_limit,
            model_path,
            self_certainty,
            confidence,
            entropy,
            apply_backoff=False
        )

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
