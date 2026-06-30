import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from prompts import get_task_instruction_math, get_task_instruction_multi_choice
from uid_evaluation import run_evaluation


SUPPORTED_DATASETS = ["aime", "brumo", "hmmt", "minervamath", "gpqa", "lsat_ar", "lsat_lr"]
MATH_DATASETS = {"aime", "brumo", "hmmt", "minervamath"}
CHOICE_DATASETS = {"gpqa", "lsat_ar", "lsat_lr"}


@dataclass
class InferenceConfig:
    seed: int
    dataset_name: str
    split: str
    subset_num: int
    model_path: str
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float | None
    max_tokens: int | None
    port: int
    batch_size: int
    data_limit: int
    sample_limit: int
    skip_special_tokens: bool
    use_beam_search: bool
    self_certainty: bool
    confidence: bool
    entropy: bool

    @classmethod
    def from_args(cls, args):
        return cls(
            seed=args.seed,
            dataset_name=args.dataset_name,
            split=args.split,
            subset_num=args.subset_num,
            model_path=args.model_path,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            max_tokens=args.max_tokens,
            port=args.port,
            batch_size=args.batch_size,
            data_limit=args.data_limit,
            sample_limit=args.sample_limit,
            skip_special_tokens=args.skip_special_tokens,
            use_beam_search=args.use_beam_search,
            self_certainty=args.self_certainty,
            confidence=args.confidence,
            entropy=args.entropy,
        )


class DatasetResolver:
    @staticmethod
    def data_path(dataset_name, split):
        if dataset_name == "aime":
            return f"../data/AIME/{split}.json"
        if dataset_name == "brumo":
            return f"../data/BRUMO/{split}.json"
        if dataset_name == "hmmt":
            return f"../data/HMMT/{split}.json"
        if dataset_name == "minervamath":
            return f"../data/MINERVA_MATH/{split}.json"
        if dataset_name == "gpqa":
            if split != "diamond":
                raise ValueError("Only the GPQA diamond split is supported. Use --split diamond.")
            return "../data/GPQA_DIAMOND_RAW/test.json"
        if dataset_name == "lsat_ar":
            return f"../data/LSAT_AR/{split}.json"
        if dataset_name == "lsat_lr":
            return f"../data/LSAT_LR/{split}.json"
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    @staticmethod
    def load(path, data_limit):
        with open(path, mode="r", encoding="utf-8") as json_file:
            data = json.load(json_file)
        return data[:data_limit]


class ModelResolver:
    @staticmethod
    def short_name(model_path):
        model_path_lower = model_path.lower()
        if "qwen-14b" in model_path_lower:
            return "ds-qwen-14b"
        if "qwen3-4b" in model_path_lower:
            return "qwen3-4b"
        if "qwen3-1.7b" in model_path_lower:
            return "qwen3-1.7b"
        if model_path_lower == "deepseek-r1-distill-qwen-7b-awq":
            return "ds-qwen-7b-awq"
        return model_path.split("/")[-1].lower().replace("-instruct", "")

    @staticmethod
    def output_dir(dataset_name, model_short_name):
        if model_short_name == "emdr2":
            return f"./outputs/{dataset_name}.{model_short_name}.direct"
        return f"./outputs/runs.baselines/{dataset_name}.{model_short_name}.direct"


class PromptBuilder:
    def __init__(self, tokenizer, model_path):
        self.tokenizer = tokenizer
        self.model_path = model_path
        self.model_path_lower = model_path.lower()

    @staticmethod
    def format_question_with_choices(item):
        question = item["Question"]
        choices = item.get("Choices")
        if not choices:
            return question

        choice_lines = []
        if isinstance(choices, dict):
            for label, text in choices.items():
                choice_lines.append(f"{label}. {text}")
        else:
            for idx, choice in enumerate(choices):
                label = chr(ord("A") + idx)
                text = choice[1] if isinstance(choice, (list, tuple)) and len(choice) > 1 else choice
                choice_lines.append(f"{label}. {text}")

        return f"{question}\n\nChoices:\n" + "\n".join(choice_lines)

    def build_user_prompt(self, item, dataset_name):
        question = item["Question"]
        if dataset_name in MATH_DATASETS:
            if any(name in self.model_path_lower for name in ["qwq", "deepseek", "sky-t1", "s1"]):
                return get_task_instruction_math(question, model_name="qwq")
            return get_task_instruction_math(question)

        if dataset_name in CHOICE_DATASETS:
            question = self.format_question_with_choices(item)
            if any(name in self.model_path_lower for name in ["qwq", "deepseek", "sky-t1"]):
                return get_task_instruction_multi_choice(question, model_name="qwq")
            if "llama" in self.model_path_lower:
                return get_task_instruction_multi_choice(question, model_name="llama")
            return get_task_instruction_multi_choice(question)

        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    def build_chat_prompt(self, item, dataset_name):
        user_prompt = self.build_user_prompt(item, dataset_name)
        prompt = [{"role": "user", "content": user_prompt}]
        return self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=True)

    def build_prompts(self, data, dataset_name):
        return [self.build_chat_prompt(item, dataset_name) for item in data]


class InferenceRunner:
    def __init__(self, config):
        self.config = config
        self.tokenizer = self._load_tokenizer()

    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return tokenizer

    def _load_llm(self):
        return LLM(
            model=self.config.model_path,
            gpu_memory_utilization=0.90,
            max_model_len=32768,
            max_num_seqs=4,
            enforce_eager=False,
            dtype="bfloat16",
            tensor_parallel_size=4,
            swap_space=32,
            trust_remote_code=True,
        )

    def _sampling_params(self):
        return SamplingParams(
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            temperature=self.config.temperature,
            max_tokens=self._max_tokens(),
            skip_special_tokens=self.config.skip_special_tokens,
            include_stop_str_in_output=True,
            logprobs=1,
        )

    def _max_tokens(self):
        max_tokens = self.config.max_tokens
        if max_tokens is None:
            max_tokens = 32768
        return min(max_tokens, 32768 - 243)

    async def _generate_outputs(self, llm, input_batches, sampling_params):
        output_list = []
        if len(input_batches) > self.config.batch_size:
            for start_index in tqdm(range(0, len(input_batches), self.config.batch_size)):
                cur_input_list = input_batches[start_index:start_index + self.config.batch_size]
                for _ in range(self.config.sample_limit):
                    batch_output = await asyncio.to_thread(llm.generate, cur_input_list, sampling_params=sampling_params)
                    output_list.extend(batch_output)
        else:
            for _ in tqdm(range(self.config.sample_limit)):
                batch_output = await asyncio.to_thread(llm.generate, input_batches, sampling_params=sampling_params)
                output_list.extend(batch_output)
        return output_list

    async def run(self):
        data_path = DatasetResolver.data_path(self.config.dataset_name, self.config.split)
        filtered_data = DatasetResolver.load(data_path, self.config.data_limit)

        prompt_builder = PromptBuilder(self.tokenizer, self.config.model_path)
        input_list = prompt_builder.build_prompts(filtered_data, self.config.dataset_name)

        if self.config.subset_num != -1:
            input_list = input_list[:self.config.subset_num]
            filtered_data = filtered_data[:self.config.subset_num]

        model_short_name = ModelResolver.short_name(self.config.model_path)
        output_dir = ModelResolver.output_dir(self.config.dataset_name, model_short_name)
        os.makedirs(output_dir, exist_ok=True)

        llm = self._load_llm()
        sampling_params = self._sampling_params()
        t_start = time.time()
        output_list = await self._generate_outputs(llm, input_list, sampling_params)
        total_time = time.time() - t_start

        run_evaluation(
            filtered_data,
            input_list,
            output_list,
            self.config.dataset_name,
            output_dir,
            total_time,
            self.config.split,
            self.config.data_limit,
            self.config.sample_limit,
            self.config.model_path,
            self.config.self_certainty,
            self.config.confidence,
            self.config.entropy,
            apply_backoff=False,
        )


def format_question_with_choices(item):
    return PromptBuilder.format_question_with_choices(item)


def parse_args():
    parser = argparse.ArgumentParser(description="Run direct generation for various datasets and models.")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--dataset_name", type=str, required=True, choices=SUPPORTED_DATASETS, help="Name of the dataset to use.")
    parser.add_argument("--split", type=str, required=True, choices=["diamond", "main", "extended", "train", "test"], help="Dataset split to use.")
    parser.add_argument("--subset_num", type=int, default=-1, help="Number of examples to process. Defaults to all if not specified.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained model.")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter.")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k sampling parameter.")
    parser.add_argument("--repetition_penalty", type=float, default=None, help="Repetition penalty. If not set, defaults based on the model.")
    parser.add_argument("--max_tokens", type=int, default=32768, help="Maximum number of tokens to generate. If not set, defaults based on the model and dataset.")
    parser.add_argument("--port", type=int, default=8100, help="Port to use for the OpenAI API.")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for the OpenAI API.")
    parser.add_argument("--data_limit", type=int, default=-1, help="Number of examples to process. Defaults to all if not specified.")
    parser.add_argument("--sample_limit", type=int, default=10, help="Number of examples to sample. Defaults to 10 if not specified.")
    parser.add_argument("--skip_special_tokens", type=bool, default=False, help="Whether to skip special tokens. Defaults to False if not specified.")
    parser.add_argument("--use_beam_search", type=bool, default=False, help="Whether to use beam search. Defaults to False if not specified.")
    parser.add_argument("--self-certainty", type=bool, default=True, help="Whether to use self-certainty. Defaults to True if not specified.")
    parser.add_argument("--confidence", type=bool, default=False, help="Whether to use confidence. Defaults to False if not specified.")
    parser.add_argument("--entropy", type=bool, default=False, help="Whether to use entropy. Defaults to False if not specified.")

    return parser.parse_args()


async def main(args):
    config = InferenceConfig.from_args(args)
    runner = InferenceRunner(config)
    await runner.run()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
