#dpo2
import json
import re
import os, time
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from evaluate_dpo2 import run_evaluation
from tqdm import tqdm
import argparse
import asyncio
import random
import numpy as np
import torch

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
        default=0.6,
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
        help="Repetition penalty. If not set, defaults based on the model."
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
    # parser.add_argument(
    #     '--step1_path',
    #     type=str,
    #     default='z_filtered',
    #     help="Path to the first stage output."
    # )

    parser.add_argument(
        '--run_type',
        type=str,
        default='test',
        choices=['test', 'full'],
        help="Whether to run test or full. Defaults to test if not specified."
    )
    return parser.parse_args()

async def main(args):
    # Set random seeds for reproducibility
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
    batch_size = args.batch_size
    sample_limit = args.sample_limit
    skip_special_tokens = args.skip_special_tokens
    use_beam_search = args.use_beam_search
    data_limit = 10
    # step1_path = args.step1_path
    run_type = args.run_type
    # Set default repetition_penalty if not provided
    if repetition_penalty is None:
        repetition_penalty = 1.05 if 'qwq' in model_path.lower() or 'deepseek' in model_path.lower() or 'sky-t1' in model_path.lower() else 1.0
    
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
            output_dir = f'./outputs/{dataset_name}.{model_short_name}.direct.step-2.{run_type}'
        else:
            output_dir = f'./outputs/runs.qa/{dataset_name}.{model_short_name}.direct.step-2.{run_type}'
    else:
        output_dir = f'./outputs/runs.baselines/{dataset_name}.{model_short_name}.direct.step-2.{run_type}'
    os.makedirs(output_dir, exist_ok=True)

    llm = LLM(
                model=model_path,
                gpu_memory_utilization=0.90,
                max_model_len=32768,
                enforce_eager=True,
                dtype="float16",
                tensor_parallel_size=4,
        )
                
    second_stage_sampling_params = SamplingParams(
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                max_tokens=max_tokens,
                repetition_penalty=repetition_penalty,
                skip_special_tokens=skip_special_tokens,
                include_stop_str_in_output=True,
        )

    
    # with open(f'./outputs/{dataset_name}.{model_short_name}.direct.step-1.{run_type}/{step1_path}.json', mode='r', encoding='utf-8') as json_file: #fix here
    #     first_stage_output_list = json.load(json_file)
    #     first_stage_output_list = first_stage_output_list[:]
    with open('./notebooks/filtered_data/step1_data.json', mode='r', encoding='utf-8') as json_file:
        first_stage_output_list = json.load(json_file)
        first_stage_output_list = first_stage_output_list[:]
        
    def clean_text(text):
        text = re.sub(r'<｜begin▁of▁sentence｜><｜User｜>', '', text, count=1)
        return text.strip()
 
    input_list = []
    new_first_stage_output_list = []
    for _, item in enumerate(first_stage_output_list):
        question = item[f'Question']
        for j in range(0, 5):
            current_first_stage_output = item[f"Output_{j}"]
            user_prompt = question + "\n\n" + current_first_stage_output
            item[f'Question_{j}'] = user_prompt

            if "free" in model_path.lower():
                prompt = [{"role": "user", "content": user_prompt}]
            else:
                prompt = [{"role": "user", "content": user_prompt}]
                prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False)

            prompt = clean_text(prompt)
            input_list.append(prompt)
            new_first_stage_output_list.append(item)

    if subset_num != -1:
        input_list = input_list[:subset_num]
        new_first_stage_output_list = new_first_stage_output_list[:subset_num]

    if max_tokens is None:
        if 'qwq' in model_path.lower() or 'deepseek' in model_path.lower() or 'sky-t1' in model_path.lower():
            if dataset_name in ['aime', 'amc', 'livecode']:
                max_tokens = 31000
            else:
                max_tokens = 31000 
        else:
            max_tokens = 31000
    max_tokens = min(max_tokens, 32768 - 243)

    async def second_stage_generate_outputs(llm, input_batches, sample_limit, batch_size):
        if len(input_batches) > batch_size:
            # Initialize a list to store responses for each input
            responses_by_input = [[] for _ in range(len(input_batches))]
            
            for start_index in tqdm(range(0, len(input_batches), batch_size)):
                cur_input_list = input_batches[start_index:start_index + batch_size]
                # Generate all samples for current batch
                for _ in tqdm(range(sample_limit)):
                    batch_output = await (asyncio.to_thread(llm.generate, cur_input_list, sampling_params=second_stage_sampling_params))
                    # Store responses for each input in the current batch
                    for i, response in enumerate(batch_output):
                        responses_by_input[start_index + i].append(response)
            
            # Interleave the responses: (response 1-1, response 2-1, ...), (response 1-2, response 2-2, ...)
            output_list = []
            for sample_idx in range(sample_limit):
                for input_responses in responses_by_input:
                    output_list.append(input_responses[sample_idx])
        else:
            output_list = []
            # Generate sample_limit outputs for each input
            for _ in tqdm(range(sample_limit)):  # Generate sample_limit times for each input
                batch_output = await (asyncio.to_thread(llm.generate, input_batches, sampling_params=second_stage_sampling_params))
                output_list.extend(batch_output)
        return output_list

    t_start = time.time()
    output_list = await second_stage_generate_outputs(llm, input_list, sample_limit, batch_size)
    # import pdb; pdb.set_trace()
    total_time = time.time() - t_start

    run_evaluation(
        new_first_stage_output_list, 
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
