#evaluate
import re
import json
import numpy as np
import string
import os, time
from collections import defaultdict
from langchain_core.outputs.chat_generation import ChatGeneration
from vllm import RequestOutput
from utils.math_equivalence import is_equiv


def extract_answer(output, mode='gen'):
    extracted_text = ''
    if mode == 'codegen':
        # Extract the code between ```python and ```
        pattern = r'```python\s*(.*?)\s*```'
        matches = re.findall(pattern, output, re.DOTALL | re.IGNORECASE)
        if matches:
            extracted_text = matches[-1].strip()  # Take the last match
    elif mode == 'infogen':
        # Extract content after **Final Information** or **Modified Reasoning Steps**
        pattern_info = "\n**Final Information**"
        pattern_step = "\n**Modified Reasoning Steps**"
        if pattern_info in output:
            extracted_text = output.split(pattern_info)[-1].replace("\n","").strip("```").strip()
        elif pattern_step in output:
            extracted_text = output.split(pattern_step)[-1].strip("```").strip()
        else:
            extracted_text = "No helpful information found."
    else:
        # Existing extraction logic for 'gen' and 'choose' modes
        pattern = r'\\boxed\{(.*)\}'
        matches = re.findall(pattern, output)
        if matches:
            extracted_text = matches[-1]  # Take the last match
            if mode in ['choose', 'qa']:
                # Handle 'choose' mode
                inner_pattern = r'\\text\{(.*)\}'
                inner_matches = re.findall(inner_pattern, extracted_text)
                if inner_matches:
                    extracted_text = inner_matches[-1]  # Take the last match
                extracted_text = extracted_text.strip("()")
    return extracted_text


def normalize_answer(text):
    text = text.lower()
    text = " ".join(text.strip().split())
    return text

def normalize_answer_qa(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.strip().split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def evaluate_predictions(output, labeled_answer, mode='gen'):
    final_metric = {"is_valid_answer": False, 'math_equal': 0}
    pred_answer = extract_answer(output, mode=mode)
    if pred_answer != '':
        final_metric["is_valid_answer"] = True

    try:
        normalized_pred_answer = normalize_answer(pred_answer)
    except:
        normalized_pred_answer = "none"

    try:
        normalized_ground_truth = normalize_answer(labeled_answer)
    except:
        normalized_ground_truth = "none"

    final_metric["math_equal"] = is_equiv(normalized_pred_answer, normalized_ground_truth)

    return final_metric, pred_answer



def run_evaluation(filtered_data, input_list, output_list, dataset_name, output_dir, total_time, split, data_limit, sample_limit, model_path, apply_backoff=False):
    # Existing evaluation for other datasets
    question_validity_scores = defaultdict(list)
    question_math_equal_scores = defaultdict(list)
    new_filtered_data = []

    for question_idx, (item, input_prompt) in enumerate(zip(filtered_data, input_list)):
        # question_samples = []
        # for i in range(question_idx, len(output_list)):
            # question_samples.append(output_list[i]) #a single row into list
        for idx in range(0, 10):
            result = output_list[question_idx]
            item_id = item["id"] + f"_{question_idx}_{idx}"
            question_key = f'Question_{question_idx}'
            answer_key = f'Answer_{question_idx}'
            output_key = f'Output_{idx}'
            tokens_key = f"output_tokens_{idx}"
            metric_key = f"Metrics_{idx}"
            pred_answer_key = f"Pred_Answer_{idx}"
            item["new_id"] = item_id
            item[question_key] = input_prompt
            item[answer_key] = item["answer"]
            if type(result) == str:
                item[output_key] = result
                item[tokens_key] = None
            elif type(result) == tuple or type(result) == list or type(result) == ChatGeneration or type(result) == RequestOutput:
                item[output_key] = result.outputs[0].text
                item[tokens_key] = len(result.outputs[0].token_ids)

            if dataset_name in ['gpqa']:
                labeled_answer = item["Correct Choice"]
                mode = 'choose'
            elif dataset_name in ['math500', 'aime', 'amc', 'hendrycks']:
                labeled_answer = item["answer"]
                mode = 'gen'
            elif dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki']:
                labeled_answer = item["answer"]
                mode = 'qa'
            else:
                raise ValueError(f"Unknown dataset_name: {dataset_name}")

            # Build the new filtered data entry
            metric, pred_answer = evaluate_predictions(output=item[output_key], labeled_answer=item[answer_key], mode=mode)
            item[pred_answer_key] = pred_answer
            item[metric_key] = metric
            
            new_item = {
                "new_id": item_id,
                question_key: item[question_key],
                answer_key: item[answer_key],
                output_key: item[output_key],
                tokens_key: item[tokens_key],
                pred_answer_key: item[pred_answer_key],
                metric_key: item[metric_key],
            }
            new_filtered_data.append(new_item)

            
            # is_valid = (pred_answer != '' and not (mode == 'choose' and dataset_name == 'gpqa' and len(pred_answer) > 1))

            question_validity_scores[question_idx].append(1 if metric['is_valid_answer'] == True else 0)
            question_math_equal_scores[question_idx].append(1 if metric['math_equal'] == True else 0)
            if idx == 0:
                item['Question'] = input_prompt

        # Compute mean accuracy and validity per question
        question_mean_validity = {}
        question_mean_math_equal = {}
        for question_idx in question_validity_scores.keys():
            question_mean_validity[f'question_{question_idx}'] = np.mean(question_validity_scores[question_idx])
            question_mean_math_equal[f'question_{question_idx}'] = np.mean(question_math_equal_scores[question_idx])

        # Compute overall metrics
        overall_results = {
            'total_time': f'{total_time:.0f} s',
            'num_valid_answer': question_mean_validity,
            'overall_validity': np.mean(list(question_validity_scores.values())),
            'num_math_equal': question_mean_math_equal,
            'overall_math_equal': np.mean(list(question_math_equal_scores.values())),
        }

        final_metrics = {
            'overall': overall_results
        }

    t = time.localtime()
    result_json_name = f'{split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}-{sample_limit}.json'
    metrics_json_name = f'{split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}-{sample_limit}.metrics.json'
    if apply_backoff:
        result_json_name = output_dir.replace('.json', f'.backoff.{sample_limit}.json')
        metrics_json_name = output_dir.replace('.json', f'.metrics.backoff.{sample_limit}.json')

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save prediction results and metrics
    with open(os.path.join(output_dir, result_json_name), mode='w', encoding='utf-8') as json_file:
        json.dump(new_filtered_data, json_file, indent=4, ensure_ascii=False)

    with open(os.path.join(output_dir, metrics_json_name), mode='w', encoding='utf-8') as json_file:
        json.dump(final_metrics, json_file, indent=4, ensure_ascii=False)



if __name__ == "__main__":
    import argparse

    # Parse command-line arguments for flexibility
    parser = argparse.ArgumentParser(description="Evaluate model outputs with optional backoff.")
    parser.add_argument('--output_path', type=str, required=True, help='Path to the model output JSON file.')
    parser.add_argument('--output_metrics_path', type=str, help='Path to save the evaluation metrics.')
    parser.add_argument('--apply_backoff', action='store_true', help='Enable backoff to normal outputs if main output is invalid.')
    parser.add_argument('--sample_limit', type=int, default=10, help='Number of samples to evaluate.')
    args = parser.parse_args()

    output_path = args.output_path
    if args.output_metrics_path:
        output_metrics_path = args.output_metrics_path
    else:
        output_metrics_path = output_path.replace('.json', '.metrics.json')

    # Determine dataset name based on the output path
    # NOTE: To apply back off strategy for retrieval-augmented reasoning methods, please replace normal_output_path with your actual path for results with run_direct_gen.
    if 'gpqa' in output_path:
        dataset_name = 'gpqa'
        normal_output_path = './outputs/gpqa.qwq.direct/diamond.12.13,18:23.json'
        if 'extended' in output_path:
            normal_output_path = './outputs/gpqa.qwq.direct/extended.12.28,15:44.json'
        if 'qwq' not in output_path:
            normal_output_path = './outputs/runs.baselines/gpqa.qwen2.5-32b-instruct.direct/diamond.12.14,20:34.json'
    elif 'math500' in output_path:
        dataset_name = 'math500'
        normal_output_path = './outputs/math500.qwq.direct/test.12.13,18:26.json'
        if 'qwq' not in output_path:
            normal_output_path = './outputs/runs.baselines/math500.qwen2.5-32b-instruct.direct/test.12.15,10:43.json'
    elif 'hendrycks' in output_path:
        dataset_name = 'hendrycks'
        normal_output_path = './outputs/runs.baselines/hendrycks.qwen2.5-32b-instruct.direct/test.12.15,10:43.json'
        if 'qwq' not in output_path:
            normal_output_path = './outputs/runs.baselines/hendrycks.qwen2.5-32b-instruct.direct/test.12.15,10:43.json'
    elif 'aime' in output_path:
        dataset_name = 'aime'
        normal_output_path = './outputs/aime.qwq.direct/2024.12.13,19:36.json'
        if 'qwq' not in output_path:
            normal_output_path = './outputs/runs.baselines/aime.qwen2.5-32b-instruct.direct/test.12.14,20:28.json'
    elif 'amc' in output_path:
        dataset_name = 'amc'
        normal_output_path = './outputs/amc.qwq.direct/test.12.14,14:31.json'
        if 'qwq' not in output_path:
            normal_output_path = './outputs/runs.baselines/amc.qwen2.5-32b-instruct.direct/test.12.14,20:26.json'
    elif 'livecode' in output_path:
        dataset_name = 'livecode'
        normal_output_path = './outputs/livecode.qwq.direct/test.12.13,21:24.json'
        if 'qwq' not in output_path:
            normal_output_path = './outputs/runs.baselines/livecode.qwen2.5-32b-instruct.direct/test.12.14,20:32.json'
    elif 'nq' in output_path:
        dataset_name = 'nq'
        normal_output_path = './outputs/runs.qa/nq.qwq.direct/test.12.15,14:50.json'
        if 'qwq' not in output_path:
            normal_output_path = ''
    elif 'triviaqa' in output_path:
        dataset_name = 'triviaqa'
        normal_output_path = './outputs/runs.qa/triviaqa.qwq.direct/test.12.15,15:35.json'
        if 'qwq' not in output_path:
            normal_output_path = ''
    elif 'hotpotqa' in output_path:
        dataset_name = 'hotpotqa'
        normal_output_path = './outputs/runs.qa/hotpotqa.qwq.direct/test.12.15,14:52.json'
        if 'qwq' not in output_path:
            normal_output_path = ''
    elif 'musique' in output_path:
        dataset_name = 'musique'
        normal_output_path = './outputs/runs.qa/musique.qwq.direct/test.12.27,16:44.json'
        if 'qwq' not in output_path:
            normal_output_path = ''
    elif 'bamboogle' in output_path:
        dataset_name = 'bamboogle'
        normal_output_path = './outputs/runs.qa/bamboogle.qwq.direct/test.12.28,9:51.json'
        if 'qwq' not in output_path:
            normal_output_path = ''
    elif '2wiki' in output_path:
        dataset_name = '2wiki'
        normal_output_path = './outputs/runs.qa/2wiki.qwq.direct/test.12.15,15:32.json'
        if 'qwq' not in output_path:
            normal_output_path = ''

    # Load main output data
    with open(output_path, mode='r', encoding='utf-8') as file:
        data = json.load(file)

    # Load main metrics data
    with open(output_metrics_path, mode='r', encoding='utf-8') as file:
        metrics = json.load(file)

    # Extract existing metrics
    if 'overall' in metrics:
        query_latency = metrics['overall']['query_latency']
        original_num_valid_answer = metrics['overall']['num_valid_answer']
        math_equal = metrics['overall']['math_equal']
    else:
        query_latency = metrics.get('query_latency', 'N/A')
        original_num_valid_answer = metrics.get('num_valid_answer', 'N/A')
        math_equal = metrics.get('math_equal', 'N/A')
    # Load normal output data if backoff is enabled
    normal_data = None
    if args.apply_backoff:
        if not os.path.exists(normal_output_path):
            raise FileNotFoundError(f"Normal output file not found at: {normal_output_path}")
        with open(normal_output_path, mode='r', encoding='utf-8') as file:
            normal_data = json.load(file)

    # Save metrics for non-livecode datasets
    if dataset_name != 'livecode' or not args.apply_backoff:
        # If dataset is livecode and backoff was applied, metrics are already saved by run_evaluation
        if args.apply_backoff:
            output_metrics_path = output_metrics_path.replace('.json', '.backoff.json')
        with open(output_metrics_path, mode='w', encoding='utf-8') as json_file:
            json.dump(final_metrics, json_file, indent=4, ensure_ascii=False)

    print(f"Evaluation completed. Metrics saved to {output_metrics_path}")
