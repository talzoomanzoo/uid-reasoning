#evaluate
import re
import json
import numpy as np
from collections import Counter
import string
import os, time
from collections import defaultdict
from lcb_runner.evaluation import codegen_metrics
from utils.math_equivalence import is_equiv
from utils.calculate_uid_rev_viz import calculate_id_metrics_with_vectors, visualize_id_vectors, visualize_average_id_vectors, visualize_average_step_counts
from utils.calculate_uid_rev_viz_self_certainty import calculate_self_certainty, calculate_borda_voting_self_certainty
from utils.calculate_uid_rev_viz_cot_decoding import calculate_cot_decoding, calculate_highest_cot_decoding
from utils.calculate_uid_rev_viz_baselines import calculate_confidence, calculate_entropy, calculate_highest_confidence, calculate_lowest_entropy
from tqdm import tqdm
from langchain_core.outputs.chat_generation import ChatGeneration
from vllm import RequestOutput

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



def run_evaluation(filtered_data, input_list, output_list, dataset_name, output_dir, total_time, split, data_limit, sample_limit, model_path, thinkseg, step_limit, self_certainty, cot_decoding, confidence, entropy, apply_backoff=False):
    if dataset_name == 'livecode':
        # Prepare samples and generations for codegen_metrics
        samples_list = []
        generations_list = []

        # Collect difficulty levels for per-domain metrics
        difficulties = []
        per_difficulty_count = {}

        #added code for medbullets, qwq-llama-distill
        output_list = [output_list.choices[i].__dict__.get('text') for i in range(len(output_list))]
        num_valid_answer = 0
        
        for item, input_prompt, result in tqdm(zip(filtered_data, input_list, output_list)):
            num_valid_answer = 0
            for i in range(len(data_limit)):
                if type(result[i]) == str:
                    item['Output'] = result[i]
                elif type(result[i]) == tuple or type(result[i]) == list:
                    item['Output'] = result[1][i][0].text
                else:
                    item['Output'] = result[i].outputs[0].text
                difficulty = item.get("difficulty", "Unknown")
                difficulties.append(difficulty)
                # Track metrics per domain
                if difficulty not in per_difficulty_count.keys():
                    per_difficulty_count[difficulty] = 0

                pred_code = extract_answer(item['Output'], mode='codegen')
                if pred_code != '':
                    num_valid_answer += 1
                    per_difficulty_count[difficulty] += 1
                # Assuming each item has 'input_output' with 'inputs' and 'outputs'
                public_test_cases = json.loads(item.get("public_test_cases", "{}"))

                inputs, outputs = [], []
                for case in public_test_cases:
                    inputs.append(case["input"])
                    outputs.append(case["output"])

                sample = {
                    "input_output": json.dumps({
                    "inputs": inputs,
                    "outputs": outputs
                }),
            }

                samples_list.append(sample)
                generations_list.append([pred_code])
                item['Pred_Answer'] = pred_code
                item['Question'] = input_prompt


        # Call codegen_metrics with pass@1
        metrics, results, final_metadata = codegen_metrics(
            samples_list,
            generations_list,
            k_list=[1],  # Evaluate the top 1 generated result
            num_process_evaluate=2,   # Parallel evaluation
            timeout=10,  # Set timeout to 10 seconds
            debug=False,  # Enable debug mode
        )
        # print('samples_list', samples_list)
        # print('generations_list', generations_list)
        # print('metrics', metrics)

        # Extract pass@1
        pass_at_1 = metrics.get('pass@1', 0.0)
        detail_pass_at_1 = metrics['detail']['pass@1']

        for item, pass1, res, meta in zip(filtered_data, detail_pass_at_1.values(), results.values(), final_metadata):
            item['Metrics'] = {'pass@1': pass1}
            item['Results'] = res
            item['Final_metadata'] = meta

        # Initialize per-difficulty metrics
        difficulty_metrics = defaultdict(list)
        for idx, difficulty in enumerate(difficulties):
            pass1 = detail_pass_at_1[idx]
            difficulty_metrics[difficulty].append(pass1)

        # Compute overall pass@1
        overall_metrics = {
            'pass@1': pass_at_1,  # / num_valid_answer * len(input_list),
            'num_valid_answer': f'{num_valid_answer} of {len(input_list)*sample_limit}',
            'query_latency': f'{(total_time / (len(input_list) * sample_limit) * 1000):.0f} s',
        }

        # Compute per-difficulty pass@1
        per_difficulty_metrics = {}
        for difficulty, passes in difficulty_metrics.items():
            avg_pass = np.mean(passes) if len(passes) > 0 else 0.0
            num_valid_answer = per_difficulty_count[difficulty]
            per_difficulty_metrics[difficulty] = {
                'pass@1': avg_pass,
                'num_valid_answer': f'{num_valid_answer} of {len(passes)*sample_limit}'
            }

        # Save the metrics
        final_metrics = {
            'overall': overall_metrics,
            'per_domain': per_difficulty_metrics
        }
        
    else:
        # Existing evaluation for other datasets
        avg_em, avg_acc, avg_f1, avg_math = [], [], [], []

        # If the dataset is GPQA or math500, track metrics per domain
        domain_metrics = {}

        # Track math_equal accuracy and validity per question
        question_math_equal_scores = defaultdict(list)
        question_validity_scores = defaultdict(list)
        question_self_certainty_scores = defaultdict(list)
        question_cot_decoding_scores = defaultdict(list)
        question_confidence_scores = defaultdict(list)
        question_entropy_scores = defaultdict(list)
        for question_idx, (item, input_prompt) in enumerate(zip(filtered_data, input_list)):
            # Get all samples for this question
            question_samples = []
            for i in range(question_idx, len(output_list), len(input_list)):
                question_samples.append(output_list[i])
                num_valid_answer = 0

            per_output_self_cert_summary = []
            per_output_cot_decoding_summary = []
            per_output_confidence_summary = []
            per_output_entropy_summary = []
            # Process each output and its metrics
            for idx in range(len(question_samples)):
                result = question_samples[idx]
                # if isinstance(result, str):
                #     item[f'Output_{idx}'] = result
                # elif isinstance(result, (tuple, list, ChatGeneration, RequestOutput)):
                item[f'Output_{idx}'] = result.outputs[0].text
                item[f"output_tokens_{idx}"] = len(result.outputs[0].token_ids)

                    
                if dataset_name in ['gpqa', 'medmcqa']:
                    labeled_answer = item["Correct Choice"]
                    mode = 'choose'
                elif dataset_name in ['math500', 'aime', 'amc', 'hendrycks', 'gsm8k', 'minervamath', 'olympiadbench', 'hmmt', 'brumo']:
                    labeled_answer = item["answer"]
                    mode = 'gen'
                elif dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki']:
                    labeled_answer = item["answer"]
                    mode = 'qa'
                elif dataset_name in ['pubhealth']:
                    labeled_answer = item["answer"]
                    mode = 'choose'
                elif dataset_name in ['medbullets', 'jama_full', 'medqa', 'medxpertqa']:
                    labeled_answer = item["answer"]
                    mode = 'choose'
                else:
                    raise ValueError(f"Unknown dataset_name: {dataset_name}")

                metric, pred_answer = evaluate_predictions(output=item[f'Output_{idx}'], labeled_answer=labeled_answer, mode=mode)
                
                # Store metrics for this specific generation
                item[f'Pred_Answer_{idx}'] = pred_answer
                item[f'Metrics_{idx}'] = metric

                sc_obj = calculate_self_certainty(result.outputs[0].logprobs)
                item[f'Self_Certainty_{idx}'] = sc_obj

                cot_decoding_obj = calculate_cot_decoding(result.outputs[0].logprobs)
                item[f'Cot_Decoding_{idx}'] = cot_decoding_obj

                confidence_obj = calculate_confidence(result.outputs[0].logprobs)
                item[f'Confidence_{idx}'] = confidence_obj

                entropy_obj = calculate_entropy(result.outputs[0].logprobs)
                item[f'Entropy_{idx}'] = entropy_obj

                per_output_self_cert_summary.append({
                    f'output_{idx}': sc_obj.get('self_certainty', float('nan')),
                    'math_equal': bool(metric.get('math_equal', False)),
                })

                per_output_cot_decoding_summary.append({
                    f'output_{idx}': cot_decoding_obj.get('confidence_score', float('nan')),
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
                highest_cot_decoding = calculate_highest_cot_decoding(per_output_cot_decoding_summary)
                highest_confidence = calculate_highest_confidence(per_output_confidence_summary)
                lowest_entropy = calculate_lowest_entropy(per_output_entropy_summary)

                is_valid = (pred_answer != '' and not (mode == 'choose' and dataset_name == 'gpqa' and len(pred_answer) > 1))

                # Track scores for this question
                if dataset_name != 'gpqa' and dataset_name != 'math500':
                    question_math_equal_scores[question_idx].append(1 if metric['math_equal'] == True else 0)
                    question_validity_scores[question_idx].append(1 if metric['is_valid_answer'] == True else 0)
                    metrics, uid_eq, uid_lp, uid_h, uid_d = calculate_id_metrics_with_vectors(result.outputs[0].logprobs, thinkseg=thinkseg)
                    item[f"id_metrics_{idx}_metrics"] = metrics
                    # Store vectors for later averaging
                    item[f"id_equal_{idx}"] = uid_eq
                    item[f"id_lp_{idx}"] = uid_lp
                    item[f"id_h_{idx}"] = uid_h
                    item[f"id_d_{idx}"] = uid_d
                    
                    # Individual visualization (optional)
                    # visualize_id_vectors(
                    #     uid_eq, uid_lp, uid_h, uid_d, dataset_name, model_path, split, 
                    #     title=f"{model_path} : ID Scores Across Steps for {dataset_name} {split}"
                    # )
                    # visualize_id_vectors(
                    #     uid_eq, uid_lp, uid_h, uid_d, dataset_name, model_path, split, title=f"ID Scores Across Steps for {dataset_name} {split}"
                    # )
                    # Store the original metrics for backward compatibility
                    if idx == 0:
                        item['Question'] = input_prompt
                    avg_math.append(metric['math_equal'])

                    if is_valid:
                        num_valid_answer += 1

                # If the dataset is GPQA, track metrics per domain
                elif dataset_name == 'gpqa':
                    domain = item.get("High-level domain", "Unknown")
                    if domain not in domain_metrics:
                        domain_metrics[domain] = {'em': [], 'acc': [], 'f1': [], 'math_equal': [], 'num_valid_answer': 0, 'total_num': 0, 'self_certainty_accuracy': [], 'cot_decoding_accuracy': [], 'confidence_accuracy': [], 'entropy_accuracy': []}
                    
                    # Add metrics for this output to the domain
                    domain_metrics[domain]['em'].append(metric['em'])
                    domain_metrics[domain]['acc'].append(metric['acc'])
                    domain_metrics[domain]['f1'].append(metric['f1'])
                    domain_metrics[domain]['math_equal'].append(metric['math_equal'])
                    domain_metrics[domain]['total_num'] += 1
                    domain_metrics[domain]['self_certainty_accuracy'].append(1 if borda_voting_self_cert['math_equal'] == True else 0)
                    domain_metrics[domain]['cot_decoding_accuracy'].append(1 if highest_cot_decoding['math_equal'] == True else 0)
                    domain_metrics[domain]['confidence_accuracy'].append(1 if highest_confidence['math_equal'] == True else 0)
                    domain_metrics[domain]['entropy_accuracy'].append(1 if lowest_entropy['math_equal'] == True else 0)
                    if idx == 0:
                        item['Question'] = input_prompt
                    
                    if is_valid:
                        domain_metrics[domain]['num_valid_answer'] += 1

                    question_math_equal_scores[question_idx].append(1 if metric['math_equal'] == True else 0)
                    question_validity_scores[question_idx].append(1 if is_valid else 0)
                    metrics, uid_eq, uid_lp, uid_h, uid_d = calculate_id_metrics_with_vectors(result.outputs[0].logprobs, thinkseg=thinkseg)
                    item[f"id_metrics_{idx}_metrics"] = metrics
                    # Store vectors for later averaging
                    item[f"id_equal_{idx}"] = uid_eq
                    item[f"id_lp_{idx}"] = uid_lp
                    item[f"id_h_{idx}"] = uid_h
                    item[f"id_d_{idx}"] = uid_d
                    
                    # Individual visualization (optional)
                    # visualize_id_vectors(
                    #     uid_eq, uid_lp, uid_h, uid_d, dataset_name, model_path, split, 
                    #     title=f"{model_path} : ID Scores Across Steps for {dataset_name} {split}"
                    # )
                    # visualize_id_vectors(
                    #     uid_eq, uid_lp, uid_h, uid_d, dataset_name, model_path, split, title=f"ID Scores Across Steps for {dataset_name} {split}"
                    # )
                
                elif dataset_name == 'math500':
                    level = item.get("level", "Unknown")
                    if level not in domain_metrics:
                        domain_metrics[level] = {'em': [], 'acc': [], 'f1': [], 'math_equal': [], 'num_valid_answer': 0, 'total_num': 0, 'self_certainty_accuracy': [], 'cot_decoding_accuracy': [], 'confidence_accuracy': [], 'entropy_accuracy': []}
                    
                    # Add metrics for this output to the level
                    domain_metrics[level]['em'].append(metric['em'])
                    domain_metrics[level]['acc'].append(metric['acc'])
                    domain_metrics[level]['f1'].append(metric['f1'])
                    domain_metrics[level]['math_equal'].append(metric['math_equal'])
                    domain_metrics[level]['total_num'] += 1
                    domain_metrics[level]['self_certainty_accuracy'].append(1 if borda_voting_self_cert['math_equal'] == True else 0)
                    domain_metrics[level]['cot_decoding_accuracy'].append(1 if highest_cot_decoding['math_equal'] == True else 0)
                    domain_metrics[level]['confidence_accuracy'].append(1 if highest_confidence['math_equal'] == True else 0)
                    domain_metrics[level]['entropy_accuracy'].append(1 if lowest_entropy['math_equal'] == True else 0)
                    if idx == 0:
                        item['Question'] = input_prompt
                    
                    if is_valid:
                        domain_metrics[level]['num_valid_answer'] += 1
                        
                    question_math_equal_scores[question_idx].append(1 if metric['math_equal'] == True else 0)
                    question_validity_scores[question_idx].append(1 if is_valid else 0)
                    metrics, uid_eq, uid_lp, uid_h, uid_d = calculate_id_metrics_with_vectors(result.outputs[0].logprobs, thinkseg=thinkseg)
                    item[f"id_metrics_{idx}_metrics"] = metrics
                    # Store vectors for later averaging
                    item[f"id_equal_{idx}"] = uid_eq
                    item[f"id_lp_{idx}"] = uid_lp
                    item[f"id_h_{idx}"] = uid_h
                    item[f"id_d_{idx}"] = uid_d
                    
                    # Individual visualization (optional)
                    # visualize_id_vectors(
                    #     uid_eq, uid_lp, uid_h, uid_d, dataset_name, model_path, split, 
                    #     title=f"{model_path} : ID Scores Across Steps for {dataset_name} {split}"
                    # )
                    # visualize_id_vectors(
                    #     uid_eq, uid_lp, uid_h, uid_d, dataset_name, model_path, split, title=f"{model_path} : ID Scores Across Steps for {dataset_name} {split}"
                    # )
                    # Visualize average ID vectors
                    # visualize_average_id_vectors(filtered_data, dataset_name, model_path, split, thinkseg)
            item['self_certainty_by_output '] = per_output_self_cert_summary
            item['borda_voting_self_cert'] = borda_voting_self_cert
            item['highest_cot_decoding'] = highest_cot_decoding
            question_self_certainty_scores[question_idx].append(1 if borda_voting_self_cert['math_equal'] == True else 0)
            question_cot_decoding_scores[question_idx].append(1 if highest_cot_decoding['math_equal'] == True else 0)
            question_confidence_scores[question_idx].append(1 if highest_confidence['math_equal'] == True else 0)
            question_entropy_scores[question_idx].append(1 if lowest_entropy['math_equal'] == True else 0)
        # Compute mean accuracy and validity per question
        if dataset_name != 'gpqa' and dataset_name != 'math500':
            question_mean_accuracies = {}
            question_upper_bound = {}
            question_mean_validities = {}
            question_self_certainty_accuracy = {}
            question_cot_decoding_accuracy = {}
            question_confidence_accuracy = {}
            question_entropy_accuracy = {}
            for question_idx in question_math_equal_scores.keys():
                question_mean_accuracies[f'question_{question_idx}'] = np.mean(question_math_equal_scores[question_idx])
                question_upper_bound[f'question_{question_idx}'] = float(np.max(question_math_equal_scores[question_idx]))
                question_mean_validities[f'question_{question_idx}'] = np.mean(question_validity_scores[question_idx])
                question_self_certainty_accuracy[f'question_{question_idx}'] = np.mean(question_self_certainty_scores[question_idx])
                question_cot_decoding_accuracy[f'question_{question_idx}'] = np.mean(question_cot_decoding_scores[question_idx])
                question_confidence_accuracy[f'question_{question_idx}'] = np.mean(question_confidence_scores[question_idx])
                question_entropy_accuracy[f'question_{question_idx}'] = np.mean(question_entropy_scores[question_idx])
                # Add per-question metrics to each item in filtered_data
            for i in range(len(filtered_data)):
                filtered_data[i]['per_question_mean_accuracy'] = question_mean_accuracies[f'question_{i}']
                filtered_data[i]['per_question_upper_bound_accuracy'] = question_upper_bound[f'question_{i}']
                filtered_data[i]['per_question_mean_validity'] = question_mean_validities[f'question_{i}']
                filtered_data[i]['per_question_mean_self_certainty_accuracy'] = question_self_certainty_accuracy[f'question_{i}']
                filtered_data[i]['per_question_mean_cot_decoding_accuracy'] = question_cot_decoding_accuracy[f'question_{i}']
                filtered_data[i]['per_question_mean_confidence_accuracy'] = question_confidence_accuracy[f'question_{i}']
                filtered_data[i]['per_question_mean_entropy_accuracy'] = question_entropy_accuracy[f'question_{i}']
            # Compute overall mean accuracy and validity across all questions
            overall_mean_accuracy = np.mean([acc for acc in question_mean_accuracies.values()])
            overall_mean_upper_bound = np.mean([ub for ub in question_upper_bound.values()])
            overall_mean_validity = np.mean([val for val in question_mean_validities.values()])
            overall_mean_self_certainty_accuracy = np.mean([acc for acc in question_self_certainty_accuracy.values()])
            overall_mean_cot_decoding_accuracy = np.mean([acc for acc in question_cot_decoding_accuracy.values()])
            overall_mean_confidence_accuracy = np.mean([acc for acc in question_confidence_accuracy.values()])
            overall_mean_entropy_accuracy = np.mean([acc for acc in question_entropy_accuracy.values()])
        else:
            question_mean_accuracies = {}
            question_upper_bound = {}
            question_mean_validities = {}
            question_self_certainty_accuracy = {}
            question_cot_decoding_accuracy = {}
            question_confidence_accuracy = {}
            question_entropy_accuracy = {}
            for question_idx in question_math_equal_scores.keys():
                question_mean_accuracies[f'question_{question_idx}'] = np.mean(question_math_equal_scores[question_idx])
                question_upper_bound[f'question_{question_idx}'] = float(np.max(question_math_equal_scores[question_idx]))
                question_mean_validities[f'question_{question_idx}'] = np.mean(question_validity_scores[question_idx])
                question_self_certainty_accuracy[f'question_{question_idx}'] = np.mean(question_self_certainty_scores[question_idx])
                question_cot_decoding_accuracy[f'question_{question_idx}'] = np.mean(question_cot_decoding_scores[question_idx])
                question_confidence_accuracy[f'question_{question_idx}'] = np.mean(question_confidence_scores[question_idx])
                question_entropy_accuracy[f'question_{question_idx}'] = np.mean(question_entropy_scores[question_idx])
            for i in range(len(filtered_data)):
                filtered_data[i]['per_question_mean_accuracy'] = question_mean_accuracies[f'question_{i}']
                filtered_data[i]['per_question_upper_bound_accuracy'] = question_upper_bound[f'question_{i}']
                filtered_data[i]['per_question_mean_validity'] = question_mean_validities[f'question_{i}']
                filtered_data[i]['per_question_mean_self_certainty_accuracy'] = question_self_certainty_accuracy[f'question_{i}']
                filtered_data[i]['per_question_mean_cot_decoding_accuracy'] = question_cot_decoding_accuracy[f'question_{i}']
                filtered_data[i]['per_question_mean_confidence_accuracy'] = question_confidence_accuracy[f'question_{i}']
                filtered_data[i]['per_question_mean_entropy_accuracy'] = question_entropy_accuracy[f'question_{i}']
                filtered_data[i]['per_question_mean_cot_decoding_accuracy'] = question_cot_decoding_accuracy[f'question_{i}']
            overall_mean_accuracy = np.mean([acc for acc in question_math_equal_scores.values()])
            overall_mean_upper_bound = np.mean([ub for ub in question_upper_bound.values()])
            overall_mean_validity = np.mean([val for val in question_validity_scores.values()])
            overall_mean_self_certainty_accuracy = np.mean([acc for acc in question_self_certainty_accuracy.values()])
            overall_mean_cot_decoding_accuracy = np.mean([acc for acc in question_cot_decoding_accuracy.values()])
            overall_mean_confidence_accuracy = np.mean([acc for acc in question_confidence_accuracy.values()])
            overall_mean_entropy_accuracy = np.mean([acc for acc in question_entropy_accuracy.values()])
        # Visualize average ID vectors across all samples
        visualize_average_id_vectors(filtered_data, dataset_name, model_path, split, thinkseg, step_limit)
        # Also visualize average step counts
        visualize_average_step_counts(filtered_data, dataset_name, model_path, split, thinkseg)

        # Compute overall metrics
        overall_results = {
            'total_time': f'{total_time:.0f} s',
            'overall_mean_accuracy': overall_mean_accuracy,  # Mean of per-question accuracies
            'overall_mean_upper_bound_accuracy': overall_mean_upper_bound,   # Mean of per-question upper bound accuracies
            'overall_mean_validity': overall_mean_validity,   # Mean of per-question validities
            'overall_mean_self_certainty_accuracy': overall_mean_self_certainty_accuracy,   # Mean of per-question self-certainty accuracy
            'overall_mean_cot_decoding_accuracy': overall_mean_cot_decoding_accuracy,   # Mean of per-question cot-decoding accuracy
            'overall_mean_confidence_accuracy': overall_mean_confidence_accuracy,   # Mean of per-question confidence accuracy
            'overall_mean_entropy_accuracy': overall_mean_entropy_accuracy,   # Mean of per-question entropy accuracy
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
                    'cot_decoding_accuracy': np.mean(m['cot_decoding_accuracy']) if len(m['cot_decoding_accuracy']) > 0 else 0,
                    'confidence_accuracy': np.mean(m['confidence_accuracy']) if len(m['confidence_accuracy']) > 0 else 0,
                    'entropy_accuracy': np.mean(m['entropy_accuracy']) if len(m['entropy_accuracy']) > 0 else 0,
                    'f1': np.mean(m['f1']) if len(m['f1']) > 0 else 0,
                    'math_equal': np.mean(m['math_equal']) if len(m['math_equal']) > 0 else 0,
                    'num_valid_answer': f'{m["num_valid_answer"]} of {m["total_num"]}',
                    'domain_mean_validity': m["num_valid_answer"] / m["total_num"],
                    'total_time': f'{total_time:.0f} s',
                }
                
        elif dataset_name == 'math500':
            for dm, m in domain_metrics.items():
                domain_avg_metrics[dm] = {
                    'upper_bound_accuracy': np.mean(m['upper_bound_accuracy']) if len(m['upper_bound_accuracy']) > 0 else 0,
                    'self_certainty_accuracy': np.mean(m['self_certainty_accuracy']) if len(m['self_certainty_accuracy']) > 0 else 0,
                    'cot_decoding_accuracy': np.mean(m['cot_decoding_accuracy']) if len(m['cot_decoding_accuracy']) > 0 else 0,
                    'confidence_accuracy': np.mean(m['confidence_accuracy']) if len(m['confidence_accuracy']) > 0 else 0,
                    'entropy_accuracy': np.mean(m['entropy_accuracy']) if len(m['entropy_accuracy']) > 0 else 0,
                    'math_equal': np.mean(m['math_equal']) if len(m['math_equal']) > 0 else 0,
                    'num_valid_answer': f'{m["num_valid_answer"]} of {m["total_num"]}',
                    'domain_mean_validity': m["num_valid_answer"] / m["total_num"],
                    'total_time': f'{total_time:.0f} s',
                }

        final_metrics = {'overall': overall_results}
        if dataset_name == 'gpqa':
            final_metrics['per_domain'] = domain_avg_metrics
        elif dataset_name == 'math500':
            final_metrics['per_domain'] = domain_avg_metrics

        t = time.localtime()
        result_json_name = f'{split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}-{sample_limit}-thinkseg{thinkseg}-step{step_limit}.json'
        metrics_json_name = f'{split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}-{sample_limit}-thinkseg{thinkseg}-step{step_limit}.metrics.json'
        if apply_backoff:
            result_json_name = output_dir.replace('.json', f'.backoff.{sample_limit}-thinkseg{thinkseg}-step{step_limit}.json')
            metrics_json_name = output_dir.replace('.json', f'.metrics.backoff.{sample_limit}-thinkseg{thinkseg}-step{step_limit}.json')

    # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # Save prediction results and metrics
        import pdb;pdb.set_trace()
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
    elif 'gsm8k' in output_path:
        dataset_name = 'gsm8k'
        normal_output_path = './outputs/gsm8k.qwq.direct/test.12.13,18:26.json'
        if 'qwq' not in output_path:
            normal_output_path = './outputs/runs.baselines/gsm8k.qwen2.5-32b-instruct.direct/test.12.15,10:43.json'
    elif 'minervamath' in output_path:
        dataset_name = 'minervamath'
        normal_output_path = './outputs/minervamath.qwq.direct/test.12.13,18:26.json'
        if 'qwq' not in output_path:
            normal_output_path = './outputs/runs.baselines/minervamath.qwen2.5-32b-instruct.direct/test.12.15,10:43.json'
    elif 'olympiadbench' in output_path:
        dataset_name = 'olympiadbench'
        normal_output_path = './outputs/olympiadbench.qwq.direct/test.12.13,18:26.json'
        if 'qwq' not in output_path:
            normal_output_path = './outputs/runs.baselines/olympiadbench.qwen2.5-32b-instruct.direct/test.12.15,10:43.json'
    elif 'hmmt' in output_path:
        dataset_name = 'hmmt'
        normal_output_path = './outputs/hmmt.qwq.direct/test.12.13,18:26.json'
        if 'qwq' not in output_path:
            normal_output_path = './outputs/runs.baselines/hmmt.qwen2.5-32b-instruct.direct/test.12.15,10:43.json'
    elif 'brumo' in output_path:
        dataset_name = 'brumo'
        normal_output_path = './outputs/brumo.qwq.direct/test.12.13,18:26.json'
        if 'qwq' not in output_path:
            normal_output_path = './outputs/runs.baselines/brumo.qwen2.5-32b-instruct.direct/test.12.15,10:43.json'
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
    elif 'medmcqa' in output_path:
        dataset_name = 'medmcqa'
        normal_output_path = './outputs/runs.qa/medmcqa.qwq.direct/test.12.15,16:57.json'
        if 'qwq' not in output_path:
            normal_output_path = ''
    elif 'pubhealth' in output_path:
        dataset_name = 'pubhealth'
        normal_output_path = './outputs/runs.qa/pubhealth.qwq.direct/test.12.15,20:32.json'
        if 'qwq' not in output_path:
            normal_output_path = ''
    #medical datasets
    elif "medbullets" in output_path:
        dataset_name = 'medbullets'
        normal_output_path = './outputs/runs.qa/medbullets.qwq.direct/test.12.15,20:32.json'
        if 'qwq' not in output_path:
            normal_output_path = ''
    elif "jama_full" in output_path:
        dataset_name = 'jama_full'
        normal_output_path = './outputs/runs.qa/jama_full.qwq.direct/test.12.15,20:32.json'
        if 'qwq' not in output_path:
            normal_output_path = ''
    elif "medqa" in output_path:
        dataset_name = 'medqa'
        normal_output_path = './outputs/runs.qa/medqa.qwq.direct/test.12.15,20:32.json'
        if 'qwq' not in output_path:
            normal_output_path = ''
    elif "medxpertqa" in output_path:
        dataset_name = 'medxpertqa'
        normal_output_path = './outputs/runs.qa/medxpertqa.qwq.direct/test.12.15,20:32.json'
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
        avg_em, avg_acc, avg_f1, avg_math = [], [], [], []
        num_valid_answer = 0

        # Initialize per-domain metrics
        domain_metrics = {}

        for i, item in enumerate(data):
            if dataset_name in ['gpqa', 'medmcqa']:
                labeled_answer = item["Correct Choice"]
                domain = item.get("High-level domain", "Unknown")
                mode = 'choose'
            elif dataset_name in ['math500', 'hendrycks']:
                labeled_answer = item["answer"]
                domain = item.get("level", "Unknown")
                mode = 'gen'
            elif dataset_name in ['aime', 'amc', 'gsm8k', 'minervamath', 'olympiadbench', 'hmmt', 'brumo']:
                labeled_answer = item["answer"]
                mode = 'gen'
                domain = 'Unknown'
            elif dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki']:
                labeled_answer = item["answer"]
                mode = 'qa'
                domain = 'Unknown'
            elif dataset_name in ['pubhealth']:
                labeled_answer = item["answer"]
                mode = 'choose'
                domain = 'Unknown'
            elif dataset_name in ['medbullets', 'jama_full', 'medqa', 'medxpertqa']:
                labeled_answer = item["answer"]
                mode = 'choose' #prompt should consider to "choose"
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
                if dataset_name in ['gpqa', 'medmcqa']:
                    normal_labeled_answer = normal_item["Correct Choice"]
                    normal_mode = 'choose'
                elif dataset_name in ['math500', 'hendrycks']:
                    normal_labeled_answer = normal_item["answer"]
                    normal_mode = 'gen'
                elif dataset_name in ['aime', 'amc', 'gsm8k', 'minervamath', 'olympiadbench', 'hmmt', 'brumo']:
                    normal_labeled_answer = normal_item["answer"]
                    normal_mode = 'gen'
                elif dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki']:
                    normal_labeled_answer = normal_item["answer"]
                    normal_mode = 'qa'
                elif dataset_name in ['pubhealth']:
                    normal_labeled_answer = normal_item["answer"]
                    normal_mode = 'choose'
                elif dataset_name in ['medbullets', 'jama_full', 'medqa', 'medxpertqa']:
                    normal_labeled_answer = normal_item["answer"]
                    normal_mode = 'choose'
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
                domain_metrics[domain] = {'em': [], 'acc': [], 'f1': [], 'math_equal': [], 'num_valid_answer': 0, 'total_num': 0, 'query_latency': [], 'upper_bound_accuracy': [], 'self_certainty_accuracy': [], 'cot_decoding_accuracy': [], 'confidence_accuracy': [], 'entropy_accuracy': []}
                domain_metrics[domain]['total_num'] += 1
                domain_metrics[domain]['query_latency'].append(item['query_latency'])
                avg_em.append(metric['em'])
                avg_acc.append(metric['acc'])
                avg_f1.append(metric['f1'])
                avg_math.append(metric['math_equal'])
                avg_upper_bound.append(1 if metric['math_equal'] == True else 0)
                avg_self_certainty.append(1 if borda_voting_self_cert['math_equal'] == True else 0)
                avg_cot_decoding.append(1 if highest_cot_decoding['math_equal'] == True else 0)
                avg_confidence.append(1 if highest_confidence['math_equal'] == True else 0)
                avg_entropy.append(1 if lowest_entropy['math_equal'] == True else 0)
                
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
            'upper_bound_accuracy': np.mean(avg_upper_bound) if len(avg_upper_bound) > 0 else 0,
            'self_certainty_accuracy': np.mean(avg_self_certainty) if len(avg_self_certainty) > 0 else 0,
            'cot_decoding_accuracy': np.mean(avg_cot_decoding) if len(avg_cot_decoding) > 0 else 0,
            'confidence_accuracy': np.mean(avg_confidence) if len(avg_confidence) > 0 else 0,
            'entropy_accuracy': np.mean(avg_entropy) if len(avg_entropy) > 0 else 0,
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
                'upper_bound_accuracy': np.mean(m['upper_bound_accuracy']) if len(m['upper_bound_accuracy']) > 0 else 0,
                'self_certainty_accuracy': np.mean(m['self_certainty_accuracy']) if len(m['self_certainty_accuracy']) > 0 else 0,
                'cot_decoding_accuracy': np.mean(m['cot_decoding_accuracy']) if len(m['cot_decoding_accuracy']) > 0 else 0,
                'confidence_accuracy': np.mean(m['confidence_accuracy']) if len(m['confidence_accuracy']) > 0 else 0,
                'entropy_accuracy': np.mean(m['entropy_accuracy']) if len(m['entropy_accuracy']) > 0 else 0,
            }

        # Prepare final metrics
        final_metrics = {'overall': overall_metrics}
        if dataset_name == 'gpqa':
            final_metrics['per_domain'] = domain_avg_metrics

    else:
        # Evaluation and backoff for livecode dataset
        split = 'test'  # Modify as needed or extract from output_path

        if args.apply_backoff and normal_data is not None:
            # Apply backoff by replacing invalid outputs with normal outputs
            for i, item in enumerate(data):
                # Extract Pred_Answer from main output
                pred_answer = item['Pred_Answer']

                # Check if Pred_Answer is invalid
                if pred_answer == '':
                    # Replace Output with normal output
                    item['Output'] = normal_data[i]['Output']

        # Prepare input_list and output_list for run_evaluation
        input_list = [item['Question'] for item in data]
        output_list = [item['Output'] for item in data]

        # Estimate total_time (if available). Here, set to 0 as a placeholder.
        total_time = 0  # Modify if timing information is available

        # Run evaluation
        run_evaluation(
            filtered_data=data,
            input_list=input_list,
            output_list=output_list,
            sample_limit=args.sample_limit,
            dataset_name=dataset_name,
            output_dir=output_path,
            total_time=total_time,
            split=split,
            self_certainty=self_certainty,
            cot_decoding=cot_decoding,
            confidence=confidence,
            entropy=entropy,
            apply_backoff=True,
        )
        # run_evaluation handles saving the metrics for livecode

    # Save metrics for non-livecode datasets
    if dataset_name != 'livecode' or not args.apply_backoff:
        # If dataset is livecode and backoff was applied, metrics are already saved by run_evaluation
        if args.apply_backoff:
            output_metrics_path = output_metrics_path.replace('.json', '.backoff-thinkseg{thinkseg}.json')
        with open(output_metrics_path, mode='w', encoding='utf-8') as json_file:
            json.dump(final_metrics, json_file, indent=4, ensure_ascii=False)

    print(f"Evaluation completed. Metrics saved to {output_metrics_path}")
