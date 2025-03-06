import argparse
import os
import json
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import re
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
from copy import deepcopy
import asyncio
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats import pointbiserialr, binned_statistic
from sklearn.metrics import roc_auc_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max_tokens", type=int, default=3000)
    parser.add_argument("--prompt_file", type=str, default="./prompts/step_level_evaluation_divided.json")
    parser.add_argument("--input_dir", type=str, default="./outputs/temp_06/expert/Sky-T1-32B-Preview/jama_full_test")
    parser.add_argument("--output_dir", type=str, default="./outputs/step_level_evaluation_divided_corrs/temp_06/Sky-T1-32B-Preview/gpt-4o-mini/jama_full_test")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--start_index", type=int, default=350)
    parser.add_argument("--final_accuracy_file", type=str, default="./outputs/temp_06/expert/Sky-T1-32B-Preview/jama_full_test/all_correct_scores.json")

    args = parser.parse_args()

    return args

def parse_stepwise_evaluation(response: str):
    count_yes = len(re.findall(r'```yes```', response, re.IGNORECASE))
    count_no = len(re.findall(r'```no```', response, re.IGNORECASE))
    return count_yes, count_no

def parse_step(response: str, step: int):
    steps = {
        1: ("Step1.", "\nStep2."),
        2: ("Step2.", "\nStep3.1."),
        3_1: ("Step3.1.", "\nStep3.2."),
        3_2: ("Step3.2.", "\nStep3.3."),
        3_3: ("Step3.3.", "\nStep4."),
        4: ("Step4.", None)
    }
    
    start, end = steps[step]
    
    # Handle optional triple hashtags
    start_pattern = f"(###?\\s*{start})"
    end_pattern = f"(###?\\s*{end})" if end else None
    
    if end:
        match = re.search(f"{start_pattern}(.*?){end_pattern}", response, re.DOTALL)
    else:
        match = re.search(f"{start_pattern}(.*)", response, re.DOTALL)
    
    if match:
        # Include the start step marker in the result
        return match.group(1) + match.group(2).strip()
    else:
        return ""

async def generate_concurrently(cur_model_data, start_index, end_index, output_dir, batch_size, model_name):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    
    llm = ChatOpenAI(
        model=model_name,
        base_url=f"https://api.openai.com/v1",
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=100,
        openai_api_key=openai_api_key
    )

    with open(args.prompt_file, "r") as f:
        prompt_data = json.load(f)
    tasks = []
    
    for i in range(start_index, end_index, batch_size):
        for j, model_data in enumerate(cur_model_data):
            output_file_path = output_dir
            if os.path.exists(os.path.join(output_file_path, f"evaluation_{start_index + j}.json")):
                print(f"Output file evaluation_{start_index + j}.json already exists, skipping...")
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
                Step1=parse_step(model_data["raw_response"], 1),
                Step2=parse_step(model_data["raw_response"], 2),
                Step3_1=parse_step(model_data["raw_response"], 3_1),
                Step3_2=parse_step(model_data["raw_response"], 3_2),
                Step3_3=parse_step(model_data["raw_response"], 3_3),
                Step4=parse_step(model_data["raw_response"], 4)
            )
            messages = [[SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]]
            response = await llm.agenerate(messages)
            token_used = response.llm_output["token_usage"]["total_tokens"]
            yes_count, no_count = parse_stepwise_evaluation(response.generations[0][0].text)
            result = deepcopy(model_data)
            result["raw_stepwise_evaluation"] = response.generations[0][0].text
            result["stepwise_evaluation_yes_count"] = yes_count
            result["stepwise_evaluation_no_count"] = no_count
            result["stepwise_evaluation_accuracy"] = yes_count / 18
            with open(os.path.join(output_file_path, f"evaluation_{idx}.json"), "w") as f:
                json.dump(result, f, indent=4)
            break
        except Exception as e:
            print(f"Error generating for {idx}: {e}")
            continue
    return result

def load_and_prepare_data(prompt_file: str, input_dir: str):
    with open(prompt_file, "r") as f:
        prompt_data = json.load(f)
    initial_model_data = []
    for file in os.listdir(input_dir):
        if file.startswith("output_"):
            with open(os.path.join(input_dir, file), "r") as f:
                data = json.load(f)
                initial_model_data.append(data)
    print(f"Loaded {len(initial_model_data)} initial model data")

    import pdb; pdb.set_trace()

    all_model_data = []
    for i in range(len(initial_model_data)):
        input_temp = dict()
        input_temp['id'] = i
        input_temp['model_input'] = prompt_data["user_input"].format(**{
            "raw_response": initial_model_data[i]["raw_response"],
            "Step1": parse_step(initial_model_data[i]["raw_response"], 1),
            "Step2": parse_step(initial_model_data[i]["raw_response"], 2),
            "Step3_1": parse_step(initial_model_data[i]["raw_response"], 3_1),
            "Step3_2": parse_step(initial_model_data[i]["raw_response"], 3_2),
            "Step3_3": parse_step(initial_model_data[i]["raw_response"], 3_3),
            "Step4": parse_step(initial_model_data[i]["raw_response"], 4)
        })
        for key in initial_model_data[i].keys():
            input_temp[key] = initial_model_data[i][key]
        all_model_data.append(input_temp)
   
    return all_model_data

async def main(args):
    all_model_data = load_and_prepare_data(args.prompt_file, args.input_dir)
    if os.path.exists(args.output_dir):
        print(f"Output directory {args.output_dir} already exists")
        return
    os.makedirs(args.output_dir)
    all_evaluation_results = []
    if len(all_model_data) > args.batch_size:
        for start_index in tqdm(range(0, len(all_model_data), args.batch_size)):
            cur_model_data = all_model_data[start_index:start_index + args.batch_size]
            print(f"Processing batch starting at index: {start_index}")
            try:
                results = await generate_concurrently(cur_model_data, start_index, start_index + args.batch_size, args.output_dir, args.batch_size, args.model_name)
                # print(f"Results from generate_concurrently: {results}")
                all_evaluation_results.extend(results)
            except Exception as e:
                print(f"Error during generate_concurrently: {e}")
    else:
        all_evaluation_results = await generate_concurrently(all_model_data, 0, len(all_model_data), args.output_dir, args.batch_size, args.model_name)

    ##correlation between average stepwise accuracy and accuracy
    # import pdb; pdb.set_trace()  # Remove or comment out the debugger line if not needed
    all_stepwise_accuracy = [result["stepwise_evaluation_accuracy"] for result in all_evaluation_results]
    avg_stepwise_accuracy = sum(all_stepwise_accuracy) / len(all_stepwise_accuracy)
    final_accuracy = json.load(open(args.final_accuracy_file, "r"))
    final_accuracy = final_accuracy[-3]
    all_final_accuracy = [float(result["is_correct"]) for result in all_evaluation_results] #change is_correct to acc
    stepwise_accuracy_with_no_as_zero = [
        0 if result["stepwise_evaluation_no_count"] > 0 else result["stepwise_evaluation_accuracy"]
        for result in all_evaluation_results
    ]

    # logistic_regression = LogisticRegression()
    # # Reshape all_stepwise_accuracy to be a 2D array
    # all_stepwise_accuracy_reshaped = np.array(all_stepwise_accuracy).reshape(-1, 1)
    # logistic_regression.fit(all_stepwise_accuracy_reshaped, all_final_accuracy)
    point_biserial_correlation = pointbiserialr(all_stepwise_accuracy, all_final_accuracy)
    point_biserial_correlation_values = [point_biserial_correlation.statistic, point_biserial_correlation.pvalue]
    auc_score = roc_auc_score(all_final_accuracy, all_stepwise_accuracy)
    #binned_accuracy = binned_statistic(all_stepwise_accuracy, all_final_accuracy, bins=10, statistic='mean')
    # import pdb; pdb.set_trace()
    
    # Create a dictionary to store the results
    results_dict = {
        "average_stepwise_accuracy": avg_stepwise_accuracy,
        "final_accuracy": final_accuracy,
        "pearson_correlation": np.corrcoef(all_stepwise_accuracy, all_final_accuracy)[0, 1],
        "pearson_correlation_with_no_as_zero": np.corrcoef(stepwise_accuracy_with_no_as_zero, all_final_accuracy)[0, 1],
        "point_biserial_correlation": point_biserial_correlation_values, #sky-t1: SignificanceResult(statistic=0.05052362508696974, pvalue=0.34252549971452834)
        "auc_score": auc_score, #sky-t1: 0.5
    }
    
    # Save the results to a JSON file
    with open(os.path.join(args.output_dir, "evaluation_results.json"), "w") as f:
        json.dump(results_dict, f, indent=4)
    
    # Optionally, print the results for verification
    print(f"Results saved to evaluation_results.json: {results_dict}")

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))