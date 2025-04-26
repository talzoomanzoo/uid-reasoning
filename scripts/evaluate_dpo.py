#evaluate
import re
import json
import numpy as np
from collections import Counter
import string
import os, time
from collections import defaultdict
from lcb_runner.evaluation import codegen_metrics
from utils.dpo_loss import score_response
from tqdm import tqdm
from langchain_core.outputs.chat_generation import ChatGeneration
from vllm import RequestOutput
from transformers import AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
# def extract_answer(output, mode='gen'):
#     extracted_text = ''
#     pattern = r'\\boxed\{(.*)\}'
#     matches = re.findall(pattern, output)
#     if matches:
#         extracted_text = matches[-1]  # Take the last match
#         if mode in ['choose', 'qa']:
#             # Handle 'choose' mode
#             inner_pattern = r'\\text\{(.*)\}'
#             inner_matches = re.findall(inner_pattern, extracted_text)
#             if inner_matches:
#                 extracted_text = inner_matches[-1]  # Take the last match
#             extracted_text = extracted_text.strip("()")
#     return extracted_text


# def normalize_answer(text):
#     text = text.lower()
#     text = " ".join(text.strip().split())
#     return text

# def normalize_answer_qa(s):
#     def remove_articles(text):
#         return re.sub(r"\b(a|an|the)\b", " ", text)
#     def white_space_fix(text):
#         return " ".join(text.strip().split())
#     def remove_punc(text):
#         exclude = set(string.punctuation)
#         return "".join(ch for ch in text if ch not in exclude)
#     def lower(text):
#         return text.lower()
#     return white_space_fix(remove_articles(remove_punc(lower(s))))

def evaluate_predictions(model, tokenizer, output, labeled_answer, mode='gen'):
    final_metric = {"is_valid_answer": False, 'bleu': 0, "rouge_1_f1": 0, "rouge_2_f1": 0, "rouge_l_f1": 0}
    # pred_answer = extract_answer(output, mode=mode)
    # if pred_answer != '':
    #     final_metric["is_valid_answer"] = True

    # if mode == 'gen':
    #     try:
    #         normalized_pred_answer = normalize_answer(pred_answer)  #predicted generation, but check the normalization function bc it isn't a single prediction
    #     except:
    #         normalized_pred_answer = "none"

    #     try:
    #         normalized_ground_truth = normalize_answer(labeled_answer) #gt solution
    #     except:
    #         normalized_ground_truth = "none"

    #     prediction_tokens = normalized_pred_answer.split()
    #     ground_truth_tokens = normalized_ground_truth.split() 
    #     common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    #     num_same = sum(common.values())
    bleu_score = sentence_bleu([labeled_answer.split()], output.split())
    rouge = Rouge()
    rouge_score = rouge.get_scores(output, labeled_answer)[0]
    final_metric["bleu"] = bleu_score
    final_metric["rouge_1_f1"] = rouge_score["rouge-1"]["f"]
    final_metric["rouge_2_f1"] = rouge_score["rouge-2"]["f"]
    final_metric["rouge_l_f1"] = rouge_score["rouge-l"]["f"]
    
    return final_metric, output



def run_evaluation(filtered_data, input_list, output_list, dataset_name, output_dir, total_time, split, data_limit, sample_limit, model_path, apply_backoff=False):
    # Existing evaluation for other datasets
    avg_bleu = []
    avg_rouge_1_f1 = []
    avg_rouge_2_f1 = []
    avg_rouge_l_f1 = []
    domain_metrics = {}

    #track mlm loss scores and validity per question
    question_bleu_scores = defaultdict(list)
    question_rouge_1_f1_scores = defaultdict(list)
    question_rouge_2_f1_scores = defaultdict(list)
    question_rouge_l_f1_scores = defaultdict(list)
    question_validity_scores = defaultdict(list)
    for question_idx, (item, input_prompt) in enumerate(zip(filtered_data, input_list)):
        question_samples = []
        for i in range(question_idx, len(output_list), len(input_list)):
            question_samples.append(output_list[i])
            num_valid_answer = 0
        for idx in range(len(question_samples)):
            result = question_samples[idx]
            if type(result) == str:
                item[f'Output_{idx}'] = result
            elif type(result) == tuple or type(result) == list or type(result) == ChatGeneration or type(result) == RequestOutput:
                item[f'Output_{idx}'] = result.outputs[0].text
                item[f"output_tokens_{idx}"] = len(result.outputs[0].token_ids)
                    
            if dataset_name in ['gpqa']:
                labeled_answer = item["Correct Choice"]
                mode = 'choose'
            elif dataset_name in ['math500', 'aime', 'amc', 'hendrycks']:
                labeled_answer = item["answer"]
                mode = 'gen'
            elif dataset_name == 's1k':
                labeled_answer = item["solution"]
                mode = 'gen'
            elif dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki']:
                labeled_answer = item["answer"]
                mode = 'qa'
            else:
                raise ValueError(f"Unknown dataset_name: {dataset_name}")

            metric, pred_answer = evaluate_predictions(model=model_path, tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True), output=item[f'Output_{idx}'], labeled_answer=labeled_answer, mode=mode)
                
            item[f'Pred_Answer_{idx}'] = pred_answer
            item[f'Metrics_{idx}'] = metric
            is_valid = (pred_answer != '' and not (mode == 'choose' and dataset_name == 'gpqa' and len(pred_answer) > 1))

            question_bleu_scores[question_idx].append(metric['bleu'])
            question_rouge_1_f1_scores[question_idx].append(metric['rouge_1_f1'])
            question_rouge_2_f1_scores[question_idx].append(metric['rouge_2_f1'])
            question_rouge_l_f1_scores[question_idx].append(metric['rouge_l_f1'])
            question_validity_scores[question_idx].append(1 if metric['is_valid_answer'] == True else 0)

            if idx == 0:
                item['Question'] = input_prompt
            avg_bleu.append(metric['bleu'])
            avg_rouge_1_f1.append(metric['rouge_1_f1'])
            avg_rouge_2_f1.append(metric['rouge_2_f1'])
            avg_rouge_l_f1.append(metric['rouge_l_f1'])

            if is_valid:
                num_valid_answer += 1

        # Compute mean accuracy and validity per question
    question_mean_bleus = {}
    question_mean_rouge_1_f1s = {}
    question_mean_rouge_2_f1s = {}
    question_mean_rouge_l_f1s = {}
    question_mean_validities = {}
    for question_idx in question_bleu_scores.keys():
        question_mean_bleus[f'question_{question_idx}'] = np.mean(question_bleu_scores[question_idx])
        question_mean_rouge_1_f1s[f'question_{question_idx}'] = np.mean(question_rouge_1_f1_scores[question_idx])
        question_mean_rouge_2_f1s[f'question_{question_idx}'] = np.mean(question_rouge_2_f1_scores[question_idx])
        question_mean_rouge_l_f1s[f'question_{question_idx}'] = np.mean(question_rouge_l_f1_scores[question_idx])
        question_mean_validities[f'question_{question_idx}'] = np.mean(question_validity_scores[question_idx])
    # Add per-question metrics to each item in filtered_data
    for i in range(len(filtered_data)):
        filtered_data[i]['per_question_mean_bleu'] = question_mean_bleus[f'question_{i}']
        filtered_data[i]['per_question_mean_rouge_1_f1'] = question_mean_rouge_1_f1s[f'question_{i}']
        filtered_data[i]['per_question_mean_rouge_2_f1'] = question_mean_rouge_2_f1s[f'question_{i}']
        filtered_data[i]['per_question_mean_rouge_l_f1'] = question_mean_rouge_l_f1s[f'question_{i}']
        filtered_data[i]['per_question_mean_validity'] = question_mean_validities[f'question_{i}']
        # Compute overall mean accuracy and validity across all questions
    overall_mean_bleu = np.mean([bleu for bleu in question_mean_bleus.values()])
    overall_mean_rouge_1_f1 = np.mean([rouge_1_f1 for rouge_1_f1 in question_mean_rouge_1_f1s.values()])
    overall_mean_rouge_2_f1 = np.mean([rouge_2_f1 for rouge_2_f1 in question_mean_rouge_2_f1s.values()])
    overall_mean_rouge_l_f1 = np.mean([rouge_l_f1 for rouge_l_f1 in question_mean_rouge_l_f1s.values()])
    overall_mean_validity = np.mean([val for val in question_mean_validities.values()])

        # Compute overall metrics
    overall_results = {
            'total_time': f'{total_time:.0f} s',
            'overall_mean_bleu': overall_mean_bleu,
            'overall_mean_rouge_1_f1': overall_mean_rouge_1_f1,   # Mean of per-question validities
            'overall_mean_rouge_2_f1': overall_mean_rouge_2_f1,
            'overall_mean_rouge_l_f1': overall_mean_rouge_l_f1,
            'overall_mean_validity': overall_mean_validity,
        }

    final_metrics = {'overall': overall_results}

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
        json.dump(filtered_data, json_file, indent=4, ensure_ascii=False)

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
    else:
        query_latency = metrics.get('query_latency', 'N/A')
        original_num_valid_answer = metrics.get('num_valid_answer', 'N/A')

    # Load normal output data if backoff is enabled
    normal_data = None
    if args.apply_backoff:
        if not os.path.exists(normal_output_path):
            raise FileNotFoundError(f"Normal output file not found at: {normal_output_path}")
        with open(normal_output_path, mode='r', encoding='utf-8') as file:
            normal_data = json.load(file)

    if dataset_name != 'livecode':
        # Existing evaluation for non-livecode datasets
        avg_bleu = []
        avg_rouge_1_f1 = []
        avg_rouge_2_f1 = []
        avg_rouge_l_f1 = []
        num_valid_answer = 0

        # Initialize per-domain metrics
        domain_metrics = {}

        for i, item in enumerate(data):
            if dataset_name == 's1k':
                labeled_answer = item["solution"]
                mode = 'gen'
                domain = 'Unknown'
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")

            output = item['Output']



            metric, pred_answer = evaluate_predictions(
                output=output, 
                labeled_answer=labeled_answer,
                mode=mode,
            )

            # Determine if the main method's answer is valid
            my_method_valid = (pred_answer != '' and not (mode == 'choose' and dataset_name == 'gpqa' and len(pred_answer) > 1))

            # If invalid and backoff is enabled, use normal method's output
            if args.apply_backoff and not my_method_valid and normal_data is not None:
                normal_item = normal_data[i]
                if dataset_name in ['gpqa']:
                    normal_labeled_answer = normal_item["Correct Choice"]
                    normal_mode = 'choose'
                elif dataset_name in ['math500', 'hendrycks']:
                    normal_labeled_answer = normal_item["answer"]
                    normal_mode = 'gen'
                elif dataset_name in ['aime', 'amc']:
                    normal_labeled_answer = normal_item["answer"]
                    normal_mode = 'gen'
                elif dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki']:
                    normal_labeled_answer = normal_item["answer"]
                    normal_mode = 'qa'
                elif dataset_name == 's1k':
                    normal_labeled_answer = normal_item["solution"]
                    normal_mode = 'gen'
                else:
                    raise ValueError(f"Unsupported dataset for backoff: {dataset_name}")

                normal_output = normal_item['Output']

                normal_metric, normal_pred_answer = evaluate_predictions(
                    output=normal_output, 
                    labeled_answer=normal_labeled_answer,
                    mode=normal_mode,
                )

                normal_valid = (normal_pred_answer != '' and not (normal_mode == 'choose' and dataset_name == 'gpqa' and len(normal_pred_answer) > 1))

                # Use normal method's result if valid
                if normal_valid:
                    metric = normal_metric
                    pred_answer = normal_pred_answer
                    my_method_valid = True

            # Track metrics per domain
            if domain not in domain_metrics:
                domain_metrics[domain] = {'bleu': [], 'rouge_1_f1': [], 'rouge_2_f1': [], 'rouge_l_f1': [], 'num_valid_answer': 0, 'total_num': 0, 'query_latency': []}
                domain_metrics[domain]['total_num'] += 1
                domain_metrics[domain]['query_latency'].append(item['query_latency'])
                avg_bleu.append(metric['bleu'])
                avg_rouge_1_f1.append(metric['rouge_1_f1'])
                avg_rouge_2_f1.append(metric['rouge_2_f1'])
                avg_rouge_l_f1.append(metric['rouge_l_f1'])

            if my_method_valid:
                num_valid_answer += 1
                domain_metrics[domain]['num_valid_answer'] += 1

        # Compute overall metrics
        overall_metrics = {
            'bleu': np.mean(avg_bleu) if len(avg_bleu) > 0 else 0, 
            'rouge_1_f1': np.mean(avg_rouge_1_f1) if len(avg_rouge_1_f1) > 0 else 0,
            'rouge_2_f1': np.mean(avg_rouge_2_f1) if len(avg_rouge_2_f1) > 0 else 0,
            'rouge_l_f1': np.mean(avg_rouge_l_f1) if len(avg_rouge_l_f1) > 0 else 0,
            'num_valid_answer': f'{num_valid_answer} of {len(data)}',
            'query_latency': query_latency,
        }
        if args.apply_backoff:
            overall_metrics['original_num_valid_answer'] = original_num_valid_answer

        # Compute per-domain metrics
        domain_avg_metrics = {}
        for dm, m in domain_metrics.items():
            domain_avg_metrics[dm] = {
                'bleu': np.mean(m['bleu']) if len(m['bleu']) > 0 else 0,
                'rouge_1_f1': np.mean(m['rouge_1_f1']) if len(m['rouge_1_f1']) > 0 else 0,
                'rouge_2_f1': np.mean(m['rouge_2_f1']) if len(m['rouge_2_f1']) > 0 else 0,
                'rouge_l_f1': np.mean(m['rouge_l_f1']) if len(m['rouge_l_f1']) > 0 else 0,
                'num_valid_answer': f'{m["num_valid_answer"]} of {m["total_num"]}',
            }

        # Prepare final metrics
        final_metrics = {'overall': overall_metrics}


    # Save metrics for non-livecode datasets
    if dataset_name != 'livecode' or not args.apply_backoff:
        # If dataset is livecode and backoff was applied, metrics are already saved by run_evaluation
        if args.apply_backoff:
            output_metrics_path = output_metrics_path.replace('.json', '.backoff.json')
        with open(output_metrics_path, mode='w', encoding='utf-8') as json_file:
            json.dump(final_metrics, json_file, indent=4, ensure_ascii=False)

    print(f"Evaluation completed. Metrics saved to {output_metrics_path}")
