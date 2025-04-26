#dpo2
import csv
import json
import random
import torch
import re
import os, time
import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from evaluate import run_evaluation
from prompts import (
    get_task_instruction_openqa, 
    get_task_instruction_math, 
    get_task_instruction_multi_choice, 
    get_task_instruction_code, 
    get_task_instruction_medical,
)
from tqdm import tqdm
import argparse
import asyncio

def parse_args():
    parser = argparse.ArgumentParser(description="Run direct generation for various datasets and models.")
    
    parser.add_argument(
        '--dataset_name',
        type=str, 
        required=True, 
        choices=['gpqa', 'math500', 'aime', 'amc', 'livecode', 'nq', 'triviaqa', 'hotpotqa', '2wiki', 'musique', 'bamboogle', 'medmcqa', 'pubhealth', 'medbullets', 'medqa', 'jama_full', 'medxpertqa'],
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
        default=0.7,
        help="Sampling temperature."
    )
    
    parser.add_argument(
        '--top_p', 
        type=float, 
        default=0.95, 
        help="Top-p sampling parameter."
    )
    
    parser.add_argument(
        '--repetition_penalty', 
        type=float, 
        default=None, 
        help="Repetition penalty. If not set, defaults based on the model."
    )
    
    parser.add_argument(
        '--max_tokens', 
        type=int, 
        default=7000, 
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
        default=10,
        help="Batch size for the OpenAI API."
    )

    parser.add_argument(
        '--sample_limit',
        type=int,
        default=4,
        help="Number of examples to sample. Defaults to 10 if not specified."
    )

    parser.add_argument(
        '--skip_special_tokens',
        type=bool,
        default=True,
        help="Whether to skip special tokens. Defaults to False if not specified."
    )

    parser.add_argument(
        '--use_beam_search',
        type=bool,
        default=False,
        help="Whether to use beam search. Defaults to False if not specified."
    )
    parser.add_argument(
        '--step1_path',
        type=str,
        default='z_filtered',
        help="Path to the first stage output."
    )
    return parser.parse_args()

async def main(args):
    dataset_name = args.dataset_name
    split = args.split
    subset_num = args.subset_num
    model_path = args.model_path
    temperature = args.temperature
    top_p = args.top_p
    repetition_penalty = args.repetition_penalty
    max_tokens = args.max_tokens
    batch_size = args.batch_size
    sample_limit = args.sample_limit
    skip_special_tokens = args.skip_special_tokens
    use_beam_search = args.use_beam_search
    data_limit = 10
    step1_path = args.step1_path
    # Set default repetition_penalty if not provided
    if repetition_penalty is None:
        repetition_penalty = 1.05 if 'qwq' in model_path.lower() or 'deepseek' in model_path.lower() or 'sky-t1' in model_path.lower() else 1.0
    
    # Paths to datasets
    if dataset_name == 'math500':
        data_path = f'../data/MATH500/{split}.json'
    elif dataset_name == 'gpqa':
        data_path = f'./data/GPQA/{split}.json'
    elif dataset_name == 'aime':
        data_path = f'./data/AIME/{split}.json'
    elif dataset_name == 'amc':
        data_path = f'./data/AMC/{split}.json'
    elif dataset_name == 'livecode':
        data_path = f'./data/LiveCodeBench/{split}.json'
    elif dataset_name in ['medbullets', 'medqa', 'jama_full', 'medxpertqa']:
        data_path = f"../data/medical/{dataset_name}_{split}.json"
    elif dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki', 'medmcqa', 'pubhealth']:
        data_path = f'./data/QA_Datasets/{dataset_name}.json'
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
    
    if 'qwq' in model_path.lower():
        model_short_name = 'qwq'
    elif 'deepseek' in model_path.lower():
        if 'qwen-14b' in model_path.lower():
            model_short_name = 'ds-qwen-14b'
        elif 'qwen-7b' in model_path.lower():
            model_short_name = 'ds-qwen-7b'
        elif 'qwen-1.5b' in model_path.lower():
            model_short_name = 'ds-qwen-1.5b'
    elif 'sky-t1' in model_path.lower():
        model_short_name = 'sky-t1'
    else:
        model_short_name = model_path.split('/')[-1].lower().replace('-instruct', '')

    if model_short_name in ['qwq', 'ds-qwen-14b', 'ds-qwen-7b', 'ds-qwen-1.5b', 'sky-t1']:
        if dataset_name in ['math500', 'gpqa', 'aime', 'amc', 'livecode']:
            output_dir = f'./outputs/{dataset_name}.{model_short_name}.direct'
        else:
            output_dir = f'./outputs/runs.qa/{dataset_name}.{model_short_name}.direct'
    else:
        output_dir = f'./outputs/runs.baselines/{dataset_name}.{model_short_name}.direct'
    os.makedirs(output_dir, exist_ok=True)

    llm = LLM(
                model=model_path,
                gpu_memory_utilization=0.90,
                max_model_len=9216,
                enforce_eager=True,
                dtype="float16",
                tensor_parallel_size=4,
        )
                
    second_stage_sampling_params = SamplingParams(
                top_p=top_p,
                temperature=temperature,
                max_tokens=max_tokens,
                repetition_penalty=repetition_penalty,
                skip_special_tokens=skip_special_tokens,
                include_stop_str_in_output=True,
        )

    # Load data
    # with open(data_path, mode='r', encoding='utf-8') as json_file:
    #     filtered_data = json.load(json_file)
    
    with open(f'./outputs/{dataset_name}.{model_short_name}.direct/{step1_path}.json', mode='r', encoding='utf-8') as json_file:
        first_stage_output_list = json.load(json_file)
        first_stage_output_list = first_stage_output_list[:]
        # first_stage_output_list = first_stage_output_list[:len(filtered_data)]
    def clean_text(text):
        text = re.sub(r'<｜begin▁of▁sentence｜><｜User｜>', '', text, count=1)
        return text.strip()
    # prepare input
    input_list = []
    output_list_1 = []
    output_list_2 = []
    for item in first_stage_output_list:
        question = item['Question']
        for j in range(0, 10):
            current_first_stage_output = item[f"Output_{j}"]
            user_prompt = question + "\n\n" + current_first_stage_output
            if "free" in model_path.lower():
                prompt = [{"role": "user", "content": user_prompt}]
            else:
                prompt = [{"role": "user", "content": user_prompt}]
                prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False)
            prompt = clean_text(prompt)
            input_list.append(prompt)
    
    if subset_num != -1:
        input_list = input_list[:subset_num]
        first_stage_output_list = first_stage_output_list[:subset_num]

    if max_tokens is None:
        if 'qwq' in model_path.lower() or 'deepseek' in model_path.lower() or 'sky-t1' in model_path.lower():
            if dataset_name in ['aime', 'amc', 'livecode']:
                max_tokens = 8000
            else:
                max_tokens = 8000 
        else:
            max_tokens = 8000
    max_tokens = min(max_tokens, 9216 - 243)

    async def second_stage_generate_outputs(llm, input_batches, model_path, max_tokens, sample_limit, temperature, top_p, batch_size):
        output_list_2 = []
        if len(input_batches) > batch_size:
            for start_index in tqdm(range(0, len(input_batches), batch_size)):
                cur_input_list = input_batches[start_index:start_index + batch_size]
                # Generate sample_limit outputs for each input in the batch
                for _ in range(sample_limit):  # Generate sample_limit times for each input
                    batch_output = await (asyncio.to_thread(llm.generate, cur_input_list, sampling_params=second_stage_sampling_params))
                    output_list_2.extend(batch_output)
        else:
            # Generate sample_limit outputs for each input
            for _ in tqdm(range(sample_limit)):  # Generate sample_limit times for each input
                batch_output = await (asyncio.to_thread(llm.generate, input_batches, sampling_params=second_stage_sampling_params))
                output_list_2.extend(batch_output)
        return output_list_2

    t_start = time.time()
    import pdb; pdb.set_trace() #check len(input_batches), batch_size
    output_list_2 = await second_stage_generate_outputs(llm, input_list, model_path, max_tokens, sample_limit, temperature, top_p, batch_size)
    total_time = time.time() - t_start

    run_evaluation(
        first_stage_output_list, 
        input_list,
        output_list_2,
        dataset_name, 
        output_dir, 
        total_time, 
        split,
        data_limit,
        sample_limit,
        model_path,
        apply_backoff=False
    )

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
