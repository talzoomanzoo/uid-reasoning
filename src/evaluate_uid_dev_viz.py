#evaluate
import re
import json
import numpy as np
from collections import Counter
import string
import os
from collections import defaultdict
from utils.math_equivalence import is_equiv, _strip_string
from utils.calculate_uid_rev_viz import calculate_id_metrics_with_vectors
from utils.calculate_uid_rev_viz_self_certainty import calculate_self_certainty, calculate_borda_voting_self_certainty
from utils.calculate_uid_rev_viz_baselines import calculate_confidence, calculate_entropy, calculate_highest_confidence, calculate_lowest_entropy
from utils.calculate_uid_rev_viz_majority_voting import calculate_majority_voting
from tqdm import tqdm

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
        pattern = r'\\boxed\{(.*)\}|ANSWER:\s*([A-D])|ANSWER:([A-D])|Answer:\s*([A-D])|Answer:([A-D])'
        matches = re.findall(pattern, output)
        if matches:
            # Take the last match and get the non-empty group
            last_match = matches[-1]
            if isinstance(last_match, tuple):
                # For patterns with multiple groups, get the non-empty one
                extracted_text = next((group for group in last_match if group), '')
            else:
                # For single group patterns, it's just a string
                extracted_text = last_match
                
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
    final_metric = {"is_valid_answer": False, "acc": 0, "em": 0, "f1": 0, 'math_equal': 0}
    
    pred_answer = extract_answer(output, mode=mode)
    if pred_answer != '':
        final_metric["is_valid_answer"] = True
    
    if mode == 'qa':
        normalized_pred_answer = normalize_answer_qa(pred_answer)
        for answer in labeled_answer:
            normalized_ground_truth = normalize_answer_qa(answer)
            em = int(normalized_pred_answer == normalized_ground_truth)
            acc = int(normalized_ground_truth in normalized_pred_answer)

            prediction_tokens = normalized_pred_answer.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            for k in ["em", "acc", "f1"]:
                final_metric[k] = max(eval(k), final_metric[k])

    else:
        try:
            normalized_pred_answer = normalize_answer(pred_answer)
        except:
            normalized_pred_answer = "none"

        try:
            normalized_ground_truth = normalize_answer(labeled_answer)
        except:
            normalized_ground_truth = "none"

        em = int(normalized_pred_answer == normalized_ground_truth)
        acc = int(normalized_ground_truth in normalized_pred_answer)
    
        prediction_tokens = normalized_pred_answer.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            f1 = 0
        else:
            precision = 1.0 * num_same / len(prediction_tokens) if len(prediction_tokens) > 0 else 0
            recall = 1.0 * num_same / len(ground_truth_tokens) if len(ground_truth_tokens) > 0 else 0
            if (precision + recall) == 0:
                f1 = 0
            else:
                f1 = (2 * precision * recall) / (precision + recall)

        final_metric["em"] = em
        final_metric["acc"] = acc
        final_metric["f1"] = f1

        final_metric["math_equal"] = is_equiv(normalized_pred_answer, normalized_ground_truth)

    # print(em, acc, f1, normalized_pred_answer, '|', normalized_ground_truth)
    return final_metric, pred_answer



def run_evaluation(filtered_data, input_list, output_list, dataset_name, output_dir, total_time, split, data_limit, sample_limit, model_path, self_certainty, confidence, entropy, apply_backoff=False):
    # Existing evaluation for other datasets
    avg_em, avg_acc, avg_f1, avg_math, avg_upper_bound, avg_self_certainty, avg_confidence, avg_entropy, avg_majority_voting = [], [], [], [], [], [], [], [], []

    # GPQA tracks metrics per high-level domain.
    domain_metrics = {}

    # Track math_equal accuracy and validity per question
    question_math_equal_scores = defaultdict(list)
    question_validity_scores = defaultdict(list)
    question_self_certainty_scores = defaultdict(list)
    question_confidence_scores = defaultdict(list)
    question_entropy_scores = defaultdict(list)
    question_majority_voting_scores = defaultdict(list)
    for question_idx, (item, input_prompt) in enumerate(zip(filtered_data, input_list)):
        # Get all samples for this question
        question_samples = []
        for i in range(question_idx, len(output_list), len(input_list)):
            question_samples.append(output_list[i])
            num_valid_answer = 0

        per_output_self_cert_summary = []
        per_output_confidence_summary = []
        per_output_entropy_summary = []
        per_output_majority_voting_summary = []
        # Process each output and its metrics
        for idx in range(len(question_samples)):
            result = question_samples[idx]
            # if isinstance(result, str):
            #     item[f'Output_{idx}'] = result
            # elif isinstance(result, (tuple, list, ChatGeneration, RequestOutput)):
            item[f'Output_{idx}'] = result.outputs[0].text
            item[f"output_tokens_{idx}"] = len(result.outputs[0].token_ids)

                
            if dataset_name in ['gpqa', 'lsat_ar', 'lsat_lr']:
                labeled_answer = item["Correct Choice"]
                mode = 'choose'
            elif dataset_name in ['aime', 'brumo', 'hmmt', 'minervamath']:
                labeled_answer = item["answer"]
                mode = 'gen'
            else:
                raise ValueError(f"Unknown dataset_name: {dataset_name}")

            metric, pred_answer = evaluate_predictions(output=item[f'Output_{idx}'], labeled_answer=labeled_answer, mode=mode)
            
            # Store metrics for this specific generation
            item[f'Pred_Answer_{idx}'] = pred_answer
            item[f'Metrics_{idx}'] = metric

            sc_obj = calculate_self_certainty(result.outputs[0].logprobs)
            item[f'Self_Certainty_{idx}'] = sc_obj


            confidence_obj = calculate_confidence(result.outputs[0].logprobs)
            item[f'Confidence_{idx}'] = confidence_obj

            entropy_obj = calculate_entropy(result.outputs[0].logprobs)
            item[f'Entropy_{idx}'] = entropy_obj
            
            majority_voting_obj = normalize_answer(extract_answer(item[f'Output_{idx}'], mode=mode))
            item[f'Majority_Voting_{idx}'] = majority_voting_obj

            per_output_majority_voting_summary.append({
                f'output_{idx}': majority_voting_obj,
                'math_equal': bool(is_equiv(majority_voting_obj, labeled_answer)),
            })

            per_output_self_cert_summary.append({
                f'output_{idx}': sc_obj.get('self_certainty', float('nan')),
                'math_equal': bool(metric.get('math_equal', False)),
            })


            per_output_confidence_summary.append({
                f'output_{idx}': confidence_obj.get('calculate_confidence', float('nan')),
                'math_equal': bool(metric.get('math_equal', False)),
            })

            per_output_entropy_summary.append({
                f'output_{idx}': entropy_obj.get('calculate_entropy', float('nan')),
                'math_equal': bool(metric.get('math_equal', False)),
            })


            borda_voting_self_cert = calculate_borda_voting_self_certainty(per_output_self_cert_summary)
            highest_confidence = calculate_highest_confidence(per_output_confidence_summary)
            lowest_entropy = calculate_lowest_entropy(per_output_entropy_summary)
            majority_voting = calculate_majority_voting(per_output_majority_voting_summary)

            is_valid = (pred_answer != '' and not (mode == 'choose' and dataset_name == 'gpqa' and len(pred_answer) > 1))

            # Track scores for this question
            if dataset_name != 'gpqa':
                question_math_equal_scores[question_idx].append(1 if metric['math_equal'] == True else 0)
                question_validity_scores[question_idx].append(1 if metric['is_valid_answer'] == True else 0)
                metrics, uid_eq, uid_lp, uid_h, uid_d = calculate_id_metrics_with_vectors(result.outputs[0].logprobs, False)
                item[f"id_metrics_{idx}_metrics"] = metrics
                # Store vectors for later averaging
                item[f"id_equal_{idx}"] = uid_eq
                item[f"id_lp_{idx}"] = uid_lp
                item[f"id_h_{idx}"] = uid_h
                item[f"id_d_{idx}"] = uid_d

                if idx == 0:
                    item['Question'] = input_prompt
                avg_math.append(metric['math_equal'])
                avg_upper_bound.append(1 if metric['math_equal'] == True else 0)
                avg_self_certainty.append(1 if borda_voting_self_cert['math_equal'] == True else 0)
                avg_confidence.append(1 if highest_confidence['math_equal'] == True else 0)
                avg_entropy.append(1 if lowest_entropy['math_equal'] == True else 0)
                avg_majority_voting.append(1 if majority_voting['math_equal'] == True else 0)
                if is_valid:
                    num_valid_answer += 1

            # If the dataset is GPQA, track metrics per domain
            elif dataset_name == 'gpqa':
                domain = item.get("High-level domain", "Unknown")
                if domain not in domain_metrics:
                    domain_metrics[domain] = {'em': [], 'acc': [], 'f1': [], 'math_equal': [], 'num_valid_answer': 0, 'total_num': 0, 'upper_bound_accuracy': [], 'self_certainty_accuracy': [], 'confidence_accuracy': [], 'entropy_accuracy': []}
                
                # Add metrics for this output to the domain
                domain_metrics[domain]['em'].append(metric['em'])
                domain_metrics[domain]['acc'].append(metric['acc'])
                domain_metrics[domain]['f1'].append(metric['f1'])
                domain_metrics[domain]['math_equal'].append(metric['math_equal'])
                domain_metrics[domain]['total_num'] += 1
                domain_metrics[domain]['upper_bound_accuracy'].append(1 if metric['math_equal'] == True else 0)
                domain_metrics[domain]['self_certainty_accuracy'].append(1 if borda_voting_self_cert['math_equal'] == True else 0)
                domain_metrics[domain]['confidence_accuracy'].append(1 if highest_confidence['math_equal'] == True else 0)
                domain_metrics[domain]['entropy_accuracy'].append(1 if lowest_entropy['math_equal'] == True else 0)
                if idx == 0:
                    item['Question'] = input_prompt
                avg_upper_bound.append(1 if metric['math_equal'] == True else 0)
                avg_self_certainty.append(1 if borda_voting_self_cert['math_equal'] == True else 0)
                avg_confidence.append(1 if highest_confidence['math_equal'] == True else 0)
                avg_entropy.append(1 if lowest_entropy['math_equal'] == True else 0)
                avg_majority_voting.append(1 if majority_voting['math_equal'] == True else 0)
                if is_valid:
                    domain_metrics[domain]['num_valid_answer'] += 1

                question_math_equal_scores[question_idx].append(1 if metric['math_equal'] == True else 0)
                question_validity_scores[question_idx].append(1 if is_valid else 0)
                metrics, uid_eq, uid_lp, uid_h, uid_d = calculate_id_metrics_with_vectors(result.outputs[0].logprobs, False)
                item[f"id_metrics_{idx}_metrics"] = metrics
                # Store vectors for later averaging
                item[f"id_equal_{idx}"] = uid_eq
                item[f"id_lp_{idx}"] = uid_lp
                item[f"id_h_{idx}"] = uid_h
                item[f"id_d_{idx}"] = uid_d
                
        item['self_certainty_by_output '] = per_output_self_cert_summary
        item['borda_voting_self_cert'] = borda_voting_self_cert
        question_self_certainty_scores[question_idx].append(1 if borda_voting_self_cert['math_equal'] == True else 0)
        question_confidence_scores[question_idx].append(1 if highest_confidence['math_equal'] == True else 0)
        question_entropy_scores[question_idx].append(1 if lowest_entropy['math_equal'] == True else 0)
        question_majority_voting_scores[question_idx].append(1 if majority_voting['math_equal'] == True else 0)
    # Compute mean accuracy and validity per question
    if dataset_name != 'gpqa':
        question_mean_accuracies = {}
        question_upper_bound = {}
        question_mean_validities = {}
        question_self_certainty_accuracy = {}
        question_confidence_accuracy = {}
        question_entropy_accuracy = {}
        question_majority_voting_accuracy = {}
        for question_idx in question_math_equal_scores.keys():
            question_mean_accuracies[f'question_{question_idx}'] = np.mean(question_math_equal_scores[question_idx])
            question_upper_bound[f'question_{question_idx}'] = float(np.max(question_math_equal_scores[question_idx]))
            question_mean_validities[f'question_{question_idx}'] = np.mean(question_validity_scores[question_idx])
            question_self_certainty_accuracy[f'question_{question_idx}'] = np.mean(question_self_certainty_scores[question_idx])
            question_confidence_accuracy[f'question_{question_idx}'] = np.mean(question_confidence_scores[question_idx])
            question_entropy_accuracy[f'question_{question_idx}'] = np.mean(question_entropy_scores[question_idx])
            question_majority_voting_accuracy[f'question_{question_idx}'] = np.mean(question_majority_voting_scores[question_idx])
            # Add per-question metrics to each item in filtered_data
        for i in range(len(filtered_data)):
            filtered_data[i]['per_question_mean_accuracy'] = question_mean_accuracies[f'question_{i}']
            filtered_data[i]['per_question_upper_bound_accuracy'] = question_upper_bound[f'question_{i}']
            filtered_data[i]['per_question_mean_validity'] = question_mean_validities[f'question_{i}']
            filtered_data[i]['per_question_mean_self_certainty_accuracy'] = question_self_certainty_accuracy[f'question_{i}']
            filtered_data[i]['per_question_mean_confidence_accuracy'] = question_confidence_accuracy[f'question_{i}']
            filtered_data[i]['per_question_mean_entropy_accuracy'] = question_entropy_accuracy[f'question_{i}']
            filtered_data[i]['per_question_majority_voting_accuracy'] = question_majority_voting_accuracy[f'question_{i}']
        # Compute overall mean accuracy and validity across all questions
        overall_mean_accuracy = np.mean([acc for acc in question_mean_accuracies.values()])
        overall_mean_upper_bound = np.mean([ub for ub in question_upper_bound.values()])
        overall_mean_validity = np.mean([val for val in question_mean_validities.values()])
        overall_mean_self_certainty_accuracy = np.mean([acc for acc in question_self_certainty_accuracy.values()])
        overall_mean_confidence_accuracy = np.mean([acc for acc in question_confidence_accuracy.values()])
        overall_mean_entropy_accuracy = np.mean([acc for acc in question_entropy_accuracy.values()])
        overall_mean_majority_voting_accuracy = np.mean([acc for acc in question_majority_voting_accuracy.values()])
    else:
        question_mean_accuracies = {}
        question_upper_bound = {}
        question_mean_validities = {}
        question_self_certainty_accuracy = {}
        question_confidence_accuracy = {}
        question_entropy_accuracy = {}
        question_majority_voting_accuracy = {}
        for question_idx in question_math_equal_scores.keys():
            question_mean_accuracies[f'question_{question_idx}'] = np.mean(question_math_equal_scores[question_idx])
            question_upper_bound[f'question_{question_idx}'] = float(np.max(question_math_equal_scores[question_idx]))
            question_mean_validities[f'question_{question_idx}'] = np.mean(question_validity_scores[question_idx])
            question_self_certainty_accuracy[f'question_{question_idx}'] = np.mean(question_self_certainty_scores[question_idx])
            question_confidence_accuracy[f'question_{question_idx}'] = np.mean(question_confidence_scores[question_idx])
            question_entropy_accuracy[f'question_{question_idx}'] = np.mean(question_entropy_scores[question_idx])
            question_majority_voting_accuracy[f'question_{question_idx}'] = np.mean(question_majority_voting_scores[question_idx])
        for i in range(len(filtered_data)):
            filtered_data[i]['per_question_mean_accuracy'] = question_mean_accuracies[f'question_{i}']
            filtered_data[i]['per_question_upper_bound_accuracy'] = question_upper_bound[f'question_{i}']
            filtered_data[i]['per_question_mean_validity'] = question_mean_validities[f'question_{i}']
            filtered_data[i]['per_question_mean_self_certainty_accuracy'] = question_self_certainty_accuracy[f'question_{i}']
            filtered_data[i]['per_question_mean_confidence_accuracy'] = question_confidence_accuracy[f'question_{i}']
            filtered_data[i]['per_question_mean_entropy_accuracy'] = question_entropy_accuracy[f'question_{i}']
            filtered_data[i]['per_question_majority_voting_accuracy'] = question_majority_voting_accuracy[f'question_{i}']
        overall_mean_accuracy = np.mean([acc for acc in question_math_equal_scores.values()])
        overall_mean_upper_bound = np.mean([ub for ub in question_upper_bound.values()])
        overall_mean_validity = np.mean([val for val in question_validity_scores.values()])
        overall_mean_self_certainty_accuracy = np.mean([acc for acc in question_self_certainty_accuracy.values()])
        overall_mean_confidence_accuracy = np.mean([acc for acc in question_confidence_accuracy.values()])
        overall_mean_entropy_accuracy = np.mean([acc for acc in question_entropy_accuracy.values()])
        overall_mean_majority_voting_accuracy = np.mean([acc for acc in question_majority_voting_accuracy.values()])
    # Compute overall metrics
    overall_results = {
        'total_time': f'{total_time:.0f} s',
        'overall_mean_accuracy': overall_mean_accuracy,  # Mean of per-question accuracies
        'overall_mean_upper_bound_accuracy': overall_mean_upper_bound,   # Mean of per-question upper bound accuracies
        'overall_mean_validity': overall_mean_validity,   # Mean of per-question validities
        'overall_mean_self_certainty_accuracy': overall_mean_self_certainty_accuracy,   # Mean of per-question self-certainty accuracy
        'overall_mean_confidence_accuracy': overall_mean_confidence_accuracy,   # Mean of per-question confidence accuracy
        'overall_mean_entropy_accuracy': overall_mean_entropy_accuracy,   # Mean of per-question entropy accuracy
        'overall_mean_majority_voting_accuracy': overall_mean_majority_voting_accuracy,   # Mean of per-question majority voting accuracy
        'overall_mean_majority_voting_accuracy': overall_mean_majority_voting_accuracy,   # Mean of per-question majority voting accuracy
    }

    # If the dataset is GPQA, output average metrics per domain
    domain_avg_metrics = {}
    if dataset_name == 'gpqa':
        for dm, m in domain_metrics.items():
            domain_avg_metrics[dm] = {
                'em': np.mean(m['em']) if len(m['em']) > 0 else 0,
                'acc': np.mean(m['acc']) if len(m['acc']) > 0 else 0,
                'upper_bound_accuracy': np.mean(m['upper_bound_accuracy']) if len(m['upper_bound_accuracy']) > 0 else 0,
                'self_certainty_accuracy': np.mean(m['self_certainty_accuracy']) if len(m['self_certainty_accuracy']) > 0 else 0,
                'confidence_accuracy': np.mean(m['confidence_accuracy']) if len(m['confidence_accuracy']) > 0 else 0,
                'entropy_accuracy': np.mean(m['entropy_accuracy']) if len(m['entropy_accuracy']) > 0 else 0,
                'majority_voting_accuracy': np.mean(m['majority_voting_accuracy']) if len(m['majority_voting_accuracy']) > 0 else 0,
                'f1': np.mean(m['f1']) if len(m['f1']) > 0 else 0,
                'math_equal': np.mean(m['math_equal']) if len(m['math_equal']) > 0 else 0,
                'num_valid_answer': f'{m["num_valid_answer"]} of {m["total_num"]}',
                'domain_mean_validity': m["num_valid_answer"] / m["total_num"],
                'total_time': f'{total_time:.0f} s',
            }
            
    final_metrics = {'overall': overall_results}
    if dataset_name == 'gpqa':
        final_metrics['per_domain'] = domain_avg_metrics
    import time
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

    # Determine dataset name based on the output path.
    # Backoff baselines are only kept for the supported datasets below.
    if 'gpqa' in output_path:
        if 'diamond' not in output_path:
            raise ValueError("Only GPQA diamond outputs are supported.")
        dataset_name = 'gpqa'
        normal_output_path = './outputs/gpqa.qwq.direct/diamond.12.13,18:23.json'
        if 'qwq' not in output_path:
            normal_output_path = './outputs/runs.baselines/gpqa.qwen2.5-32b-instruct.direct/diamond.12.14,20:34.json'
    elif 'aime' in output_path:
        dataset_name = 'aime'
        normal_output_path = './outputs/aime.qwq.direct/2024.12.13,19:36.json'
        if 'qwq' not in output_path:
            normal_output_path = './outputs/runs.baselines/aime.qwen2.5-32b-instruct.direct/test.12.14,20:28.json'
    elif 'brumo' in output_path:
        dataset_name = 'brumo'
        normal_output_path = './outputs/brumo.qwq.direct/test.12.13,18:26.json'
        if 'qwq' not in output_path:
            normal_output_path = './outputs/runs.baselines/brumo.qwen2.5-32b-instruct.direct/test.12.15,10:43.json'
    elif 'hmmt' in output_path:
        dataset_name = 'hmmt'
        normal_output_path = './outputs/hmmt.qwq.direct/test.12.13,18:26.json'
        if 'qwq' not in output_path:
            normal_output_path = './outputs/runs.baselines/hmmt.qwen2.5-32b-instruct.direct/test.12.15,10:43.json'
    elif 'minervamath' in output_path:
        dataset_name = 'minervamath'
        normal_output_path = './outputs/minervamath.qwq.direct/test.12.13,18:26.json'
        if 'qwq' not in output_path:
            normal_output_path = './outputs/runs.baselines/minervamath.qwen2.5-32b-instruct.direct/test.12.15,10:43.json'
    elif 'lsat_ar' in output_path:
        dataset_name = 'lsat_ar'
        normal_output_path = ''
    elif 'lsat_lr' in output_path:
        dataset_name = 'lsat_lr'
        normal_output_path = ''
    else:
        raise ValueError(f"Unsupported dataset output path: {output_path}")

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

    if True:
        # Existing evaluation for supported datasets
        avg_em, avg_acc, avg_f1, avg_math = [], [], [], []
        num_valid_answer = 0

        # Initialize per-domain metrics
        domain_metrics = {}

        for i, item in enumerate(data):
            if dataset_name in ['gpqa', 'lsat_ar', 'lsat_lr']:
                labeled_answer = item["Correct Choice"]
                domain = item.get("High-level domain", "Unknown")
                mode = 'choose'
            elif dataset_name in ['aime', 'brumo', 'hmmt', 'minervamath']:
                labeled_answer = item["answer"]
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

            avg_em.append(metric['em'])
            avg_acc.append(metric['acc'])
            avg_f1.append(metric['f1'])
            avg_math.append(metric['math_equal'])

            if my_method_valid:
                num_valid_answer += 1

            # If invalid and backoff is enabled, use normal method's output
            if args.apply_backoff and not my_method_valid and normal_data is not None:
                normal_item = normal_data[i]
                if dataset_name in ['gpqa', 'lsat_ar', 'lsat_lr']:
                    normal_labeled_answer = normal_item["Correct Choice"]
                    normal_mode = 'choose'
                elif dataset_name in ['aime', 'brumo', 'hmmt', 'minervamath']:
                    normal_labeled_answer = normal_item["answer"]
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

                avg_em.append(normal_metric['em'])
                avg_acc.append(normal_metric['acc'])
                avg_f1.append(normal_metric['f1'])
                avg_math.append(normal_metric['math_equal'])

                # Use normal method's result if valid
                if normal_valid:
                    metric = normal_metric
                    pred_answer = normal_pred_answer
                    my_method_valid = True

            # Track metrics per domain
            if domain not in domain_metrics:
                domain_metrics[domain] = {'em': [], 'acc': [], 'f1': [], 'math_equal': [], 'num_valid_answer': 0, 'total_num': 0, 'query_latency': [], 'upper_bound_accuracy': [], 'self_certainty_accuracy': [], 'confidence_accuracy': [], 'entropy_accuracy': []}
                domain_metrics[domain]['total_num'] += 1
                domain_metrics[domain]['query_latency'].append(item['query_latency'])
                avg_em.append(metric['em'])
                avg_acc.append(metric['acc'])
                avg_f1.append(metric['f1'])
                avg_math.append(metric['math_equal'])
                
            if my_method_valid:
                num_valid_answer += 1
                domain_metrics[domain]['num_valid_answer'] += 1

        # Compute overall metrics
        overall_metrics = {
            'em': np.mean(avg_em) if len(avg_em) > 0 else 0, 
            'acc': np.mean(avg_acc) if len(avg_acc) > 0 else 0, 
            'f1': np.mean(avg_f1) if len(avg_f1) > 0 else 0, 
            'math_equal': np.mean(avg_math) if len(avg_math) > 0 else 0, 
            'num_valid_answer': f'{num_valid_answer} of {len(data)}',
            'query_latency': query_latency,
        }
        if args.apply_backoff:
            overall_metrics['original_num_valid_answer'] = original_num_valid_answer

        # Compute per-domain metrics
        domain_avg_metrics = {}
        for dm, m in domain_metrics.items():
            domain_avg_metrics[dm] = {
                'em': np.mean(m['em']) if len(m['em']) > 0 else 0,
                'acc': np.mean(m['acc']) if len(m['acc']) > 0 else 0,
                'f1': np.mean(m['f1']) if len(m['f1']) > 0 else 0,
                'math_equal': np.mean(m['math_equal']) if len(m['math_equal']) > 0 else 0,
                'num_valid_answer': f'{m["num_valid_answer"]} of {m["total_num"]}',
            }

        # Prepare final metrics
        final_metrics = {'overall': overall_metrics}
        if dataset_name == 'gpqa':
            final_metrics['per_domain'] = domain_avg_metrics

    if args.apply_backoff:
        output_metrics_path = output_metrics_path.replace('.json', '.backoff.json')
    with open(output_metrics_path, mode='w', encoding='utf-8') as json_file:
        json.dump(final_metrics, json_file, indent=4, ensure_ascii=False)

    print(f"Evaluation completed. Metrics saved to {output_metrics_path}")