import os
import asyncio
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from tqdm.asyncio import tqdm_asyncio
import json
from tqdm import tqdm
import argparse
from copy import deepcopy
import re
from utils.evaluate import evaluate_prediction
from utils.average import average_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--input_file", type=str, default="../data/medical/jama_full_test.json")
    parser.add_argument("--prompt_file", type=str, default="./prompts/naive.json")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--start_index", type=int, default=0)

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"./outputs/naive/{args.model_name.split('/')[-1].split('.')[0]}/{args.input_file.split('/')[-1].split('.')[0]}"
    return args

def parse_response(response: str):
    match = re.search(r'````?\s*[Oo]ption?\s*[{]?\s*([A-D])?\s*[{]?\s*```', response)
    return match.group(1) if match else None

async def generate_concurrently(cur_model_data, start_index, end_index, output_dir, batch_size, model_name):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    llm = ChatOpenAI(
        model=model_name,
        base_url="http://localhost:8100/v1",
        temperature=0,
        max_tokens=9000,
        max_retries=100,
        openai_api_key=openai_api_key
    )
    
    # Load prompt data once
    with open(args.prompt_file, "r") as f:
        prompt_data = json.load(f)
    tasks = []
    
    # Process data in batches
    for i in range(start_index, end_index, batch_size):
        for j, model_data in enumerate(cur_model_data):
            output_file_path = output_dir
            if os.path.exists(os.path.join(output_file_path, f"output_{start_index + j}.json")):
                print(f"Output file output_{start_index + j}.json already exists, skipping...")
                continue
            tasks.append(async_generate(llm, model_data, output_file_path, start_index + j, prompt_data))
    if tasks:
        return await tqdm_asyncio.gather(*tasks)
    else:
        return []

async def async_generate(llm, model_data, output_file_path, idx, prompt_data):
    system_prompt = prompt_data["system_prompt"]
    while True:
        try:
            user_prompt = prompt_data["user_input"].format(
            question=model_data["Question"],
            opa=model_data["opa"],
            opb=model_data["opb"],
            opc=model_data["opc"],
            opd=model_data["opd"]
            )
            messages = [[SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]]
            # import pdb; pdb.set_trace()
            response = await llm.agenerate(messages)
            # import pdb; pdb.set_trace()
            token_used = response.llm_output["token_usage"]["total_tokens"]
            result = deepcopy(model_data)
            result["raw_response"] = response.generations[0][0].text
            result["token_used"] = token_used
            result["prediction"] = parse_response(response.generations[0][0].text)
            result["is_correct"] = evaluate_prediction(result["prediction"], result["answer_idx"])
            with open(os.path.join(output_file_path, f"output_{idx}.json"), "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            break
        except Exception as e:
            print(f"Error: {e}")
            continue
    return result

def load_and_prepare_data(prompt_file: str, input_file: str):
    with open(prompt_file, "r") as f:
        prompt_data = json.load(f)
    with open(input_file, "r") as f:
        data = json.load(f)
    all_model_data = []
    for i in range(len(data)):
        input_temp = dict()
        input_temp['id'] = i
        input_temp['model_input'] = prompt_data["user_input"].format(**{
            "question": data[i]["Question"],
            "opa": data[i]["opa"],
            "opb": data[i]["opb"],
            "opc": data[i]["opc"],
            "opd": data[i]["opd"]
        })
        for key in data[i].keys():
            input_temp[key] = data[i][key]
        all_model_data.append(input_temp)
    return all_model_data
    
async def main(args):
    all_model_data = load_and_prepare_data(args.prompt_file, args.input_file)
    if os.path.exists(args.output_dir):
        print(f"Output directory {args.output_dir} already exists")
        return
    os.makedirs(args.output_dir)
    all_results = []
    if len(all_model_data) > args.batch_size:
        for start_index in tqdm(range(0, len(all_model_data), args.batch_size)):
            cur_model_data = all_model_data[start_index:start_index + args.batch_size]
            print(f"Processing batch starting at index: {start_index}")
            try:
                results = await generate_concurrently(cur_model_data, start_index, start_index + args.batch_size, args.output_dir, args.batch_size, args.model_name)
                print(f"Results from generate_concurrently: {results}")
                all_results.extend(results)
            except Exception as e:
                print(f"Error during generate_concurrently: {e}")
    else:
        all_results = await generate_concurrently(all_model_data, 0, len(all_model_data), args.output_dir, args.batch_size, args.model_name)

    all_correct_scores = [result["is_correct"] for result in all_results]
    avg_score = average_score(all_correct_scores)
    print(f"Average score: {avg_score}")
    all_correct_scores.append(avg_score)
    try:
        with open(os.path.join(args.output_dir, "all_correct_scores.json"), "w") as f: #not correctly implemented
            json.dump(all_correct_scores, f, indent=4, ensure_ascii=False)
        print("File saved successfully.")
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))