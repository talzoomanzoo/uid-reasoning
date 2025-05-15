import json
import random
import torch
import os, time
import numpy as np
from collections import defaultdict
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from evaluate_refactor import run_evaluation
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
        '--seed',
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )
    
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
        default=31000, 
        help="Maximum number of tokens to generate. If not set, defaults based on the model and dataset."
    )

    parser.add_argument(
        '--port',
        type=int,
        default=8100,
        help="Port to use for the OpenAI API."
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

    # parser.add_argument(
    #     '--batch_size',
    #     type=int,
    #     default=1,
    #     help="Batch size. Defaults to 1 if not specified."
    # )

    return parser.parse_args()

async def main(args):
    #set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset_name = args.dataset_name
    split = args.split
    subset_num = args.subset_num
    model_path = args.model_path
    temperature = args.temperature
    top_p = args.top_p
    top_k = args.top_k
    repetition_penalty = args.repetition_penalty
    max_tokens = args.max_tokens
    data_limit = args.data_limit
    sample_limit = args.sample_limit
    skip_special_tokens = args.skip_special_tokens
    # batch_size = args.batch_size
    # Set default repetition_penalty if not provided
    if repetition_penalty is None:
        repetition_penalty = 1.05 if 'qwq' in model_path.lower() or 'deepseek' in model_path.lower() or 'sky-t1' in model_path.lower() else 1.0
    
    # Paths to datasets
    if dataset_name == 'math500':
        data_path = f'../data/MATH500/{split}.json'
    elif dataset_name == 'gpqa':
        data_path = f'../data/GPQA/{split}.json'
    elif dataset_name == 'aime':
        data_path = f'../data/AIME/{split}.json'

    elif dataset_name == 'amc':
        data_path = f'../data/AMC/{split}.json'
    elif dataset_name == 'livecode':
        data_path = f'../data/LiveCodeBench/{split}.json'
    elif dataset_name in ['medbullets', 'medqa', 'jama_full', 'medxpertqa']:
        data_path = f"../data/medical/{dataset_name}_{split}.json"
    elif dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki', 'medmcqa', 'pubhealth']:
        data_path = f'../data/QA_Datasets/{dataset_name}.json'
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")
    
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
                max_model_len=32768,
                enforce_eager=True,
                dtype="float16",
                tensor_parallel_size=4,
        )
                
    sampling_params = SamplingParams(
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                max_tokens=max_tokens,
                repetition_penalty=repetition_penalty,
                skip_special_tokens=skip_special_tokens,
                include_stop_str_in_output=True,
        )


    # Load data
    with open(data_path, mode='r', encoding='utf-8') as json_file:
        filtered_data = json.load(json_file)
        filtered_data = filtered_data[:data_limit]
    
    # prepare input
    input_list = []
    for item in filtered_data:
        question = item['Question']
        if dataset_name in ['medbullets', 'medqa', 'jama_full', 'medxpertqa']:
            options = [item['opa'], item['opb'], item['opc'], item['opd']]
            if 'qwq' in model_path.lower() or 'deepseek' in model_path.lower() or 'sky-t1' in model_path.lower():
                user_prompt = get_task_instruction_medical(question, options, model_name='qwq')
            elif 'llama' in model_path.lower():
                user_prompt = get_task_instruction_medical(question, model_name='llama')
            else:
                user_prompt = get_task_instruction_medical(question)

        elif dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki']:
            if 'qwq' in model_path.lower() or 'deepseek' in model_path.lower() or 'sky-t1' in model_path.lower():
                user_prompt = get_task_instruction_openqa(question, model_name='qwq')
            else:
                user_prompt = get_task_instruction_openqa(question)

        elif dataset_name in ['math500', 'aime', 'amc']:
            if 'qwq' in model_path.lower() or 'deepseek' in model_path.lower() or 'sky-t1' in model_path.lower() or 's1' in model_path.lower():
                user_prompt = get_task_instruction_math(question, model_name='qwq')
            else:
                user_prompt = get_task_instruction_math(question)

        elif dataset_name in ['gpqa']:
            if 'qwq' in model_path.lower() or 'deepseek' in model_path.lower() or 'sky-t1' in model_path.lower():
                user_prompt = get_task_instruction_multi_choice(question, model_name='qwq')
            elif 'llama' in model_path.lower():
                user_prompt = get_task_instruction_multi_choice(question, model_name='llama')
            else:
                user_prompt = get_task_instruction_multi_choice(question)
            
        elif dataset_name == 'livecode':
            question_title = item.get('question_title', '')
            if 'qwq' in model_path.lower() or 'deepseek' in model_path.lower() or 'sky-t1' in model_path.lower():
                user_prompt = get_task_instruction_code(question, question_title=question_title, model_name='qwq')
            else:
                user_prompt = get_task_instruction_code(question)
        else:
            user_prompt = ""  # Default to empty if dataset not matched

        if "free" in model_path.lower():
            prompt = [{"role": "user", "content": user_prompt}]
        else:
            prompt = [{"role": "user", "content": user_prompt}]
            prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        input_list.append(prompt)
    
    if subset_num != -1:
        input_list = input_list[:subset_num]
        filtered_data = filtered_data[:subset_num]
    
    # Set default max_tokens if not provided
    if max_tokens is None:
        if 'qwq' in model_path.lower() or 'deepseek' in model_path.lower() or 'sky-t1' in model_path.lower():
            if dataset_name in ['aime', 'amc', 'livecode']:
                max_tokens = 31000
            else:
                max_tokens = 31000 
        else:
            max_tokens = 31000

    # Adjust max_tokens to fit within the model's context length
    max_tokens = min(max_tokens, 32768 - 243)  # Ensure total tokens do not exceed 4096

    # async def generate_outputs_prev(llm, input_batches, sample_limit, batch_size):
    #     output_list = []
    #     if len(input_batches) > batch_size:
    #         for start_index in tqdm(range(0, len(input_batches), batch_size)):
    #             cur_input_list = input_batches[start_index:start_index + batch_size]
    #             # Generate sample_limit outputs for each input in the batch
    #             for _ in range(sample_limit):  # Generate sample_limit times for each input
    #                 batch_output = await (asyncio.to_thread(llm.generate, cur_input_list, sampling_params=sampling_params))
    #                 output_list.extend(batch_output)
    #     else:
    #         # Generate sample_limit outputs for each input
    #         for _ in tqdm(range(sample_limit)):  # Generate sample_limit times for each input
    #             batch_output = await (asyncio.to_thread(llm.generate, input_batches, sampling_params=sampling_params))
    #             output_list.extend(batch_output)
    #     return output_list

    # async def generate_outputs_wo_order(llm, input_batches, sample_limit):
    #     all_prompts = []
    #     for prompt in input_batches:
    #         all_prompts.extend([prompt] * sample_limit)

    #     output_list = await asyncio.to_thread(
    #         llm.generate,
    #         all_prompts,
    #         sampling_params=sampling_params
    #     )
    #     return output_list

    async def generate_outputs(llm, input_batches, sample_limit):
    # Step 1: Attach indices to each prompt
        indexed_prompts = []
        index_to_prompt = []

        for idx, prompt in enumerate(tqdm(input_batches)):
            for _ in range(sample_limit):
                indexed_prompts.append(prompt)
                index_to_prompt.append(idx)  # Track which input this belongs to

        # Step 2: Generate outputs in bulk
        output_list = await asyncio.to_thread(
            llm.generate,
            indexed_prompts,
            sampling_params=sampling_params
        )

        # Step 3: Re-group outputs by original input index
        grouped_outputs = defaultdict(list)
        for i, output in enumerate(tqdm(output_list)):
            orig_idx = index_to_prompt[i]
            grouped_outputs[orig_idx].append(output)

        # Step 4: Return outputs grouped by input_batches order
        return [grouped_outputs[i] for i in range(len(input_batches))]

    t_start = time.time()
    output_list = await generate_outputs(llm, input_list, sample_limit)
    total_time = time.time() - t_start
    
    # Run evaluation
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
        apply_backoff=False
    )

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
