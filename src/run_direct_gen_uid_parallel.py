import csv
import json
import random
import torch
import re
import os, time
import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from evaluate_uid_parallel import run_evaluation
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
from typing import List, Dict, Any, Iterable, Tuple

# ----------------------------
# argparse with robust flags
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run direct generation (batched) and evaluation (batched + final).")

    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")

    parser.add_argument(
        '--dataset_name', type=str, required=True,
        choices=['gpqa','math500','aime','amc','livecode','nq','triviaqa','hotpotqa','2wiki','musique','bamboogle',
                 'medmcqa','pubhealth','medbullets','medqa','jama_full','medxpertqa'],
        help="Dataset to use."
    )
    parser.add_argument(
        '--split', type=str, required=True,
        choices=['diamond','main','extended','train','test'],
        help="Dataset split."
    )
    parser.add_argument('--subset_num', type=int, default=-1, help="Use only first N examples (after filtering).")
    parser.add_argument('--model_path', type=str, required=True, help="Path/hub id of the model.")
    parser.add_argument('--temperature', type=float, default=0.6, help="Sampling temperature.")
    parser.add_argument('--top_p', type=float, default=0.95, help="Top-p sampling.")
    parser.add_argument('--top_k', type=int, default=20, help="Top-k sampling.")
    parser.add_argument('--repetition_penalty', type=float, default=None, help="Repetition penalty (default 1.05).")
    parser.add_argument('--max_tokens', type=int, default=32768, help="Max new tokens.")
    parser.add_argument('--port', type=int, default=8100, help="(unused) Port.")
    parser.add_argument('--batch_size', type=int, default=100, help="Batch size for prompts.")
    parser.add_argument('--data_limit', type=int, default=-1, help="Global data cap before subset_num.")
    parser.add_argument('--sample_limit', type=int, default=10, help="Number of samples per prompt.")
    parser.add_argument('--skip_special_tokens', action='store_true', help="Skip special tokens in outputs.")
    parser.add_argument('--use_beam_search', action='store_true', help="(unused) Beam search toggle.")

    # evaluation batching controls
    parser.add_argument('--eval_per_batch', action='store_true', help="Run evaluation for each batch.")
    parser.add_argument('--save_intermediate', action='store_true', help="Write JSONL outputs per batch to disk.")
    return parser.parse_args()

# ----------------------------
# helpers
# ----------------------------
def chunked(seq: List[Any], size: int) -> Iterable[Tuple[int, List[Any]]]:
    """Yield (start_index, chunk) pairs."""
    for i in range(0, len(seq), size):
        yield i, seq[i:i+size]

def infer_data_path(dataset_name: str, split: str) -> str:
    if dataset_name == 'math500':
        return f'../data/MATH500/{split}.json'
    if dataset_name == 'gpqa':
        return f'../data/GPQA/{split}.json'
    if dataset_name == 'aime':
        return f'../data/AIME/{split}.json'
    if dataset_name == 'amc':
        return f'../data/AMC/{split}.json'
    if dataset_name == 'livecode':
        return f'./data/LiveCodeBench/{split}.json'
    if dataset_name in ['medbullets','medqa','jama_full','medxpertqa']:
        return f"../data/medical/{dataset_name}_{split}.json"
    if dataset_name in ['nq','triviaqa','hotpotqa','musique','bamboogle','2wiki','medmcqa','pubhealth']:
        return f'./data/QA_Datasets/{dataset_name}.json'
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")

def short_model_name(model_path: str) -> str:
    mp = model_path.lower()
    if 'qwen-14b' in mp: return 'ds-qwen-14b'
    if 'qwen3-4b' in mp: return 'qwen3-4b'
    if 'qwen3-1.7b' in mp: return 'qwen3-1.7b'
    if mp == 'deepseek-r1-distill-qwen-7b-awq': return 'ds-qwen-7b-awq'
    if 'sft-gt' in mp: return 'ds-qwen-7b-sft-gt'
    if 'rft-y-lora-y-true' in mp: return 'ds-qwen-7b-rft-y-lora-y-true'
    if 'rft-z-lora-y-true' in mp: return 'ds-qwen-7b-rft-z-lora-y-true'
    if 'rft-z-lora-z-threshold' in mp: return 'ds-qwen-7b-rft-z-lora-z-threshold'
    if 'dpo-y-lora-y-true' in mp: return 'ds-qwen-7b-dpo-y-lora-y-true'
    if 'dpo-zy-lora-y-true' in mp: return 'ds-qwen-7b-dpo-zy-lora-y-true'
    if 'dpo-zy-lora-z-threshold' in mp: return 'ds-qwen-7b-dpo-zy-lora-z-threshold'
    if 'sky-t1' in mp: return 'sky-t1'
    return model_path.split('/')[-1].lower().replace('-instruct', '')

def build_output_dir(dataset_name: str, model_short_name: str) -> str:
    if model_short_name in ['emdr2']:
        base = './outputs' if dataset_name in ['math500','gpqa','aime','amc','livecode'] else './outputs/runs.qa'
        out = f'{base}/{dataset_name}.{model_short_name}.direct'
    else:
        out = f'./outputs/runs.baselines/{dataset_name}.{model_short_name}.direct'
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out, "batches"), exist_ok=True)
    return out

def get_med_options(item: Dict[str, Any]) -> List[str]:
    # supports opa..ope if present
    keys = ['opa','opb','opc','opd','ope']
    return [item[k] for k in keys if k in item]

def make_user_prompt(dataset_name: str, model_path: str, item: Dict[str, Any]) -> str:
    q = item['Question']
    mp = model_path.lower()
    is_qwq_like = any(k in mp for k in ['qwq','deepseek','sky-t1'])
    if dataset_name in ['medbullets','medqa','jama_full','medxpertqa']:
        options = get_med_options(item)
        if is_qwq_like:
            return get_task_instruction_medical(q, options, model_name='qwq')
        elif 'llama' in mp:
            return get_task_instruction_medical(q, options, model_name='llama')
        else:
            return get_task_instruction_medical(q, options)
    elif dataset_name in ['nq','triviaqa','hotpotqa','musique','bamboogle','2wiki']:
        return get_task_instruction_openqa(q, model_name='qwq' if is_qwq_like else None)
    elif dataset_name in ['math500','aime','amc']:
        return get_task_instruction_math(q, model_name='qwq' if is_qwq_like or 's1' in mp else None)
    elif dataset_name == 'gpqa':
        if is_qwq_like:
            return get_task_instruction_multi_choice(q, model_name='qwq')
        elif 'llama' in mp:
            return get_task_instruction_multi_choice(q, model_name='llama')
        else:
            return get_task_instruction_multi_choice(q)
    elif dataset_name == 'livecode':
        qt = item.get('question_title', '')
        return get_task_instruction_code(q, question_title=qt, model_name='qwq' if is_qwq_like else None)
    else:
        return ""

async def main(args):
    # ----------------------------
    # seeds & defaults
    # ----------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    repetition_penalty = 1.05 if args.repetition_penalty is None else args.repetition_penalty
    max_tokens = min(args.max_tokens or 32768, 32768 - 243)

    data_path = infer_data_path(args.dataset_name, args.split)

    # ----------------------------
    # tokenizer (only for chat template)
    # ----------------------------
    if "free" in args.model_path.lower():
        tokenizer = None
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'

    model_short = short_model_name(args.model_path)
    output_dir = build_output_dir(args.dataset_name, model_short)

    # ----------------------------
    # model (sized for batching)
    # ----------------------------
    llm = LLM(
        model=args.model_path,
        gpu_memory_utilization=0.65,
        max_model_len=32768,
        max_num_seqs=max(4, args.batch_size),   # allow full batches
        enforce_eager=False,
        dtype="bfloat16",
        tensor_parallel_size=4,
        swap_space=32,
    )

    # NB: We keep n=1 and generate sample_limit *times* to preserve the original
    # shape (N * sample_limit RequestOutputs) expected by your evaluation code.
    base_sampling = dict(
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
        skip_special_tokens=args.skip_special_tokens,
        include_stop_str_in_output=False,
        logprobs=1,
    )

    # ----------------------------
    # load & cap data
    # ----------------------------
    with open(data_path, 'r', encoding='utf-8') as f:
        full_data = json.load(f)

    if args.data_limit != -1:
        full_data = full_data[:args.data_limit]
    if args.subset_num != -1:
        full_data = full_data[:args.subset_num]

    # ----------------------------
    # prompts (keep 1:1 with data)
    # ----------------------------
    input_prompts: List[str] = []
    for item in full_data:
        up = make_user_prompt(args.dataset_name, args.model_path, item)
        if tokenizer is not None:
            chat = [{"role": "user", "content": up}]
            # keep enable_thinking if your tokenizer supports it; it's ignored otherwise
            prompt = tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True, enable_thinking=True
            )
        else:
            prompt = up
        input_prompts.append(prompt)

    # ----------------------------
    # generation BY BATCH, evaluation BY BATCH,
    # and FINAL gathering at the end
    # ----------------------------
    all_outputs = []  # flattened list of RequestOutput objects, same as your original shape

    t0 = time.time()
    batch_metrics = []  # optional storage if run_evaluation returns metrics (not required)

    # pre-split to keep consistent indices
    batches = list(chunked(input_prompts, args.batch_size))

    for b_idx, (start, prompts_batch) in enumerate(tqdm(batches, desc="Batches")):
        # align data & prompts for evaluation per batch
        data_batch = full_data[start:start + len(prompts_batch)]
        inputs_batch = prompts_batch

        # generate sample_limit times in parallel for this batch (keeps shape identical to your original)
        async def gen_once(s_iter: int):
            sp = SamplingParams(**base_sampling, seed=(args.seed + s_iter))
            # vLLM generate is sync; run in a thread to parallelize across s_iter
            return await asyncio.to_thread(llm.generate, inputs_batch, sampling_params=sp)

        # launch all sample passes together and preserve order
        gen_tasks = [asyncio.create_task(gen_once(s)) for s in range(args.sample_limit)]
        gen_results_lists = await asyncio.gather(*gen_tasks)  # list of [RequestOutput,...] * sample_limit

        # flatten in (sample pass major) order to match your previous code
        batch_outputs = []
        for res in gen_results_lists:
            batch_outputs.extend(res)

        all_outputs.extend(batch_outputs)

        # save intermediates if requested (useful for crash recovery)
        if args.save_intermediate:
            # write JSONL with minimal fields to avoid serializing the whole objects if undesired
            os.makedirs(os.path.join(output_dir, "batches"), exist_ok=True)
            jpath = os.path.join(output_dir, "batches", f"batch_{b_idx:04d}.jsonl")
            with open(jpath, "w", encoding="utf-8") as jf:
                for ro in batch_outputs:
                    # ro: vllm.outputs.RequestOutput; serialize essentials
                    jf.write(json.dumps({
                        "request_id": ro.request_id,
                        "prompt": ro.prompt,
                        "outputs": [{"text": oo.text} for oo in ro.outputs],
                    }, ensure_ascii=False) + "\n")

        # optional: evaluate by batch
        if args.eval_per_batch:
            try:
                run_evaluation(
                    data_batch,
                    inputs_batch,
                    batch_outputs,
                    args.dataset_name,
                    os.path.join(output_dir, f"batch_{b_idx:04d}"),
                    total_time=0.0,  # per-batch time not critical; you can time each if you want
                    split=args.split,
                    data_limit=len(data_batch),
                    sample_limit=args.sample_limit,
                    model_path=args.model_path,
                    apply_backoff=False
                )
            except Exception as e:
                print(f"[warn] Batch {b_idx} evaluation failed: {e}")

    total_time = time.time() - t0

    # FINAL gather: evaluate once on the whole run
    run_evaluation(
        full_data,
        input_prompts,
        all_outputs,
        args.dataset_name,
        output_dir,
        total_time,
        args.split,
        len(full_data),
        args.sample_limit,
        args.model_path,
        apply_backoff=False
    )

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
