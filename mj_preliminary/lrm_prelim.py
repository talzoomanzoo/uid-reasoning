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
from utils.average import average_score, average_token_used

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--input_file", type=str, default="../data/medical/jama_full_test.json")
    parser.add_argument("--prompt_file", type=str, default="./prompts/naive.json")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--temperature", type=float, default=0.6)

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"./outputs/naive/{args.model_name.split('/')[-1].split('.')[0]}/{args.input_file.split('/')[-1].split('.')[0]}"
    return args

def parse_response(response: str):
    match = re.search(r'```([A-D])```', response)
    return match.group(1) if match else None

async def generate_concurrently(cur_model_data, start_index, end_index, output_dir, batch_size, model_name):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    
    openrouter_api_key = os.getenv("OPEN_ROUTER_API_KEY")
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY is not set")
    
    if "gpt" in model_name:
        llm = ChatOpenAI(
            model=model_name,
            base_url=f"https://api.openai.com/v1",
            temperature=args.temperature,
            max_tokens=3000,
            max_retries=100,
            openai_api_key=openai_api_key
        )
    elif "gemini" in model_name:
        llm = ChatOpenAI(
            model=model_name,
            base_url=f"https://openrouter.ai/api/v1",
            temperature=args.temperature,
            max_tokens=3000,
            max_retries=100,
            api_key=openrouter_api_key
        )
    else:
        llm = ChatOpenAI(
            model=model_name,
        base_url=f"http://localhost:{args.port}/v1",
        temperature=args.temperature,
        max_tokens=3000,
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
            response = await llm.agenerate(messages)
            token_used = response.llm_output["token_usage"]["total_tokens"]
            result = deepcopy(model_data)
            result["raw_response"] = response.generations[0][0].text
            result["prediction"] = parse_response(response.generations[0][0].text)
            result["is_correct"] = evaluate_prediction(result["prediction"], result["answer_idx"])
            if result["is_correct"] == True:
                result["correct_token_used"] = token_used
            else:
                result["incorrect_token_used"] = token_used
            with open(os.path.join(output_file_path, f"output_{idx}.json"), "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            break
        except Exception as e:
            print(f"Error: {e}")
            continue
    return result

def load_and_prepare_step_level_evaluate_data(raw_response: str):
    with open("./prompts/step_level_evaluate.json", "r") as f:
        prompt_data = json.load(f)
    return prompt_data["user_input"].format(**{
        "raw_response": raw_response
        })

def load_and_prepare_data(prompt_file: str, input_file: str):
    #main data
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
                # print(f"Results from generate_concurrently: {results}")
                all_results.extend(results)
            except Exception as e:
                print(f"Error during generate_concurrently: {e}")
    else:
        all_results = await generate_concurrently(all_model_data, 0, len(all_model_data), args.output_dir, args.batch_size, args.model_name)

    all_correct_scores = [result["is_correct"] for result in all_results]
    correct_token_used = 0
    incorrect_token_used = 0
    correct_num = 0
    incorrect_num = 0
    for result in all_results:
        if result["is_correct"] == True:
            correct_token_used += result["correct_token_used"]
            correct_num += 1
        else:
            incorrect_token_used += result["incorrect_token_used"]
            incorrect_num += 1
    avg_score = average_score(all_correct_scores)
    avg_correct_token_used = average_token_used(correct_token_used, correct_num)
    avg_incorrect_token_used = average_token_used(incorrect_token_used, incorrect_num)
    print(f"Average score: {avg_score}")
    print(f"Average correct token used: {avg_correct_token_used}")
    print(f"Average incorrect token used: {avg_incorrect_token_used}")
    all_correct_scores.append(avg_score)
    all_correct_scores.append(avg_correct_token_used)
    all_correct_scores.append(avg_incorrect_token_used)
    try:
        with open(os.path.join(args.output_dir, "all_correct_scores.json"), "w") as f:
            json.dump(all_correct_scores, f, indent=4, ensure_ascii=False)
        print("File saved successfully.")
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))