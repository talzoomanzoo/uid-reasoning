from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

#initial configs
model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
prompt_prefix = "Convert the point $(0,3)$ in rectangular coordinates to polar coordinates. Enter your answer in the form $(r,\theta),$ where $r > 0$ and $0 \le \theta < 2 \pi.$ The solution is"
prompt = [{"role": "user", "content": prompt_prefix}]
prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
stop_token_1 = "</think>"

#first stage sampling
llm = LLM(
    model=model_path,
    gpu_memory_utilization=0.90,
    max_model_len=8192,
    enforce_eager=True,
    dtype="float16",
    tensor_parallel_size=4,
    )


first_stage_sampling_params = SamplingParams(
    top_p=0.95,
    temperature=0.6,
    max_tokens=7000,
    skip_special_tokens=False,
    include_stop_str_in_output=False,
    stop=stop_token_1
)

print("Sampling first-stage (reflection) completions...")
first_stage_outputs = llm.generate(prompt, first_stage_sampling_params)
first_stage_samples = [o.outputs[0].text.strip() for o in first_stage_outputs]

# Display first-stage thoughts
for i, sample in enumerate(first_stage_samples):
    print(f"\n--- First Stage Sample {i+1} ---")
    print(prompt + sample + stop_token_1)



#second stage sampling
second_stage_sampling_params = SamplingParams(
    top_p=0.95,
    temperature=0.6,
    max_tokens=7000,
    skip_special_tokens=False,
    include_stop_str_in_output=True,
)

all_final_completions = []

print("\nSampling second-stage (final answer) completions...")
for i, partial_thought in enumerate(first_stage_samples):
    full_prompt = prompt + partial_thought + stop_token_1
    second_stage_outputs = llm.generate(full_prompt, second_stage_sampling_params)

    completions = [o.outputs[0].text.strip() for o in second_stage_outputs]
    all_final_completions.append((full_prompt, completions))

# === Display All Results ===
for i, (full_prompt, completions) in enumerate(all_final_completions):
    # print(f"\n====== From First-Stage Sample {i+1} ======")
    # print(f"Prompt used:\n{full_prompt}\n")
    for j, comp in enumerate(completions):
        print(f"Second-Stage Sample {j+1}: {comp}")
